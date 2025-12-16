#!/usr/bin/env julia

using ArgParse
using FASTX.FASTA
using RestrictedBoltzmannMachines: load_rbm, sample_from_inputs, sample_v_from_v, gpu, cpu
using CUDA
using ProgressMeter
using Random

function parse_commandline()
    s = ArgParseSettings(description = "Generate sequences from a trained RBM model and save them as FASTA file.")
    
    @add_arg_table! s begin
        "--rbm_model_path"
            help = "Path to the RBM model file (.h5)."
            required = true
        "--output"
            help = "Path to the output FASTA file."
            required = true
        "--num_sequences"
            help = "Number of sequences to generate."
            arg_type = Int
            default = 1000
        "--num_steps"
            help = "Number of Gibbs sampling steps to run."
            arg_type = Int
            default = 1000
        "--batch_size"
            help = "Batch size for generating sequences (for memory efficiency)."
            arg_type = Int
            default = 2000
        "--use_gpu"
            help = "Use GPU for generation if available."
            action = :store_true
        "--seed"
            help = "Random seed for reproducibility."
            arg_type = Int
            default = nothing
    end
    
    return parse_args(s)
end

function decode_onehot_to_sequences(onehot_data::AbstractArray, tokens::Vector{Char})
    """
    Decode one-hot encoded sequences back to string sequences.
    
    Args:
        onehot_data: One-hot encoded array of shape (ntokens, seqlen, nseqs)
        tokens: Vector of amino acid tokens
    
    Returns:
        Vector of protein sequences as strings
    """
    ntokens, seqlen, nseqs = size(onehot_data)
    sequences = Vector{String}(undef, nseqs)
    
    for i in 1:nseqs
        seq_chars = Vector{Char}(undef, seqlen)
        for j in 1:seqlen
            # Find the token with maximum value (should be 1 in one-hot)
            token_idx = argmax(onehot_data[:, j, i])
            seq_chars[j] = tokens[token_idx]
        end
        sequences[i] = String(seq_chars)
    end
    
    return sequences
end

function generate_sequences(rbm_model, num_sequences::Int, num_steps::Int, batch_size::Int, use_gpu::Bool)
    """
    Generate sequences from RBM model using Gibbs sampling.
    
    Args:
        rbm_model: Loaded RBM model
        num_sequences: Total number of sequences to generate
        num_steps: Number of Gibbs sampling steps
        batch_size: Batch size for generation
        use_gpu: Whether to use GPU
    
    Returns:
        Generated sequences as one-hot encoded array of shape (ntokens, seqlen, num_sequences)
    """
    println("Generating $num_sequences sequences with $num_steps Gibbs sampling steps...")
    
    # Move model to GPU if requested
    if use_gpu && CUDA.functional()
        println("Using GPU for generation")
        rbm_gpu = gpu(rbm_model)
    else
        println("Using CPU for generation")
        rbm_gpu = rbm_model
    end
    
    # Get dimensions from the model
    ntokens, seqlen, nhidden = size(rbm_gpu.w)
    
    # Initialize storage for all generated sequences
    all_generated = zeros(Bool, ntokens, seqlen, num_sequences)
    
    # Generate in batches
    num_batches = ceil(Int, num_sequences / batch_size)
    
    @showprogress "Generating sequences: " for batch_idx in 1:num_batches
        # Calculate batch size for this iteration
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, num_sequences)
        current_batch_size = end_idx - start_idx + 1
        
        # Initialize sequences from the model's visible layer distribution with zeros
        if use_gpu && CUDA.functional()
            sampled_v = sample_from_inputs(rbm_gpu.visible, gpu(zeros(ntokens, seqlen, current_batch_size)))
        else
            sampled_v = sample_from_inputs(rbm_gpu.visible, zeros(ntokens, seqlen, current_batch_size))
        end
        
        # Run Gibbs sampling
        for step in 1:num_steps
            sampled_v = sample_v_from_v(rbm_gpu, sampled_v; steps=1)
        end
        
        # Store generated sequences
        all_generated[:, :, start_idx:end_idx] = cpu(sampled_v)
    end
    
    return all_generated
end

function save_sequences_to_fasta(sequences::Vector{String}, output_path::String)
    """
    Save sequences to a FASTA file.
    
    Args:
        sequences: Vector of protein sequences
        output_path: Path to output FASTA file
    """
    println("Writing sequences to $output_path...")
    
    FASTA.Writer(open(output_path, "w")) do writer
        for (i, seq) in enumerate(sequences)
            # Create a simple header with sequence index
            header = "generated_seq_$i"
            record = FASTA.Record(header, seq)
            write(writer, record)
        end
    end
    
    println("Successfully wrote $(length(sequences)) sequences to $output_path")
end

function main()
    # Parse command line arguments
    args = parse_commandline()
    
    # Set random seed if provided
    if args["seed"] !== nothing
        Random.seed!(args["seed"])
        println("Random seed set to $(args["seed"])")
    end
    
    # Define amino acid alphabet (must match the one used during training)
    tokens = collect("-ACDEFGHIKLMNPQRSTVWY")  # 20 amino acids + gap
    println("Using alphabet: $(String(tokens))")
    
    # Load RBM model
    println("Loading RBM model from $(args["rbm_model_path"])...")
    rbm_model = load_rbm(args["rbm_model_path"])
    
    # Print model information
    ntokens, seqlen, nhidden = size(rbm_model.w)
    println("RBM model dimensions:")
    println("  - Number of tokens: $ntokens")
    println("  - Sequence length: $seqlen")
    println("  - Hidden units: $nhidden")
    
    # Generate sequences
    generated_onehot = generate_sequences(
        rbm_model,
        args["num_sequences"],
        args["num_steps"],
        args["batch_size"],
        args["use_gpu"]
    )
    
    # Decode to string sequences
    println("Decoding sequences...")
    generated_sequences = decode_onehot_to_sequences(generated_onehot, tokens)
    
    # Save to FASTA file
    save_sequences_to_fasta(generated_sequences, args["output"])
    
    println("Generation complete!")
end

# Run main function
main()
