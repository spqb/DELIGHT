using ArgParse
using CSV
using DataFrames
using PyCall
using FASTX.FASTA
using FASTX: identifier, sequence
using RestrictedBoltzmannMachines: load_rbm, sample_h_from_v, gpu, cpu
using CUDA
using ProgressMeter
using HDF5

function parse_commandline()
    s = ArgParseSettings(description = "Encodes aligned sequences using RBM embedding and saves them as npz files.")
    
    @add_arg_table! s begin
        "--input"
            help = "Path to the input dataset in .csv format."
            required = true
        "--rbm_model_path"
            help = "Path to the RBM model file (.h5)."
            required = true
        "--column_sequences"
            help = "Column name in the input .csv file containing the sequences."
            default = "sequence_align"
        "--column_labels"
            help = "Column name in the input .csv file containing the labels."
            default = "label"
        "--column_headers"
            help = "Column name in the input .csv file containing the sequence identifiers."
            default = "header"
        "--use_gpu"
            help = "Use GPU for encoding if available."
            action = :store_true
        "--batch_size"
            help = "Batch size for encoding sequences. If not specified, processes all sequences at once."
            arg_type = Int
            default = 0
    end
    
    return parse_args(s)
end

function encode_sequences_to_onehot(sequences::Vector{String}, tokens::Vector{Char})
    """
    Encode sequences to one-hot representation.
    
    Args:
        sequences: Vector of aligned protein sequences (all same length)
        tokens: Vector of amino acid tokens
    
    Returns:
        One-hot encoded array of shape (ntokens, seqlen, nseqs)
    """
    ntokens = length(tokens)
    nseqs = length(sequences)
    seqlen = length(sequences[1])
    
    # Create token to index mapping
    token_to_int = Dict{Char, Int}()
    for (i, token) in enumerate(tokens)
        token_to_int[token] = i
    end
    
    # Encode to one-hot
    msa_onehot = zeros(Bool, ntokens, seqlen, nseqs)
    for i in 1:nseqs
        for j in 1:seqlen
            token_idx = token_to_int[sequences[i][j]]
            msa_onehot[token_idx, j, i] = true
        end
    end
    
    return msa_onehot
end

function rbm_encode_sequences(sequences::Vector{String}, rbm_model, tokens::Vector{Char}, use_gpu::Bool, batch_size::Int=0)
    """
    RBM encode a list of aligned sequences.
    
    Args:
        sequences: Vector of aligned protein sequences (all same length)
        rbm_model: Loaded RBM model
        tokens: Vector of amino acid tokens
        use_gpu: Whether to use GPU for encoding
        batch_size: Batch size for processing. If 0, processes all sequences at once.
    
    Returns:
        RBM encoded array of shape (nseqs, nhidden)
    """
    nseqs = length(sequences)
    
    # If batch_size is 0 or greater than nseqs, process all at once
    if batch_size <= 0 || batch_size >= nseqs
        println("Encoding sequences to one-hot...")
        msa_onehot = encode_sequences_to_onehot(sequences, tokens)
        
        println("Computing RBM hidden layer activations...")
        if use_gpu && CUDA.functional()
            rbm_gpu = gpu(rbm_model)
            msa_gpu = gpu(msa_onehot)
            hidden = sample_h_from_v(rbm_gpu, msa_gpu)
            hidden_cpu = cpu(hidden)
        else
            hidden_cpu = sample_h_from_v(rbm_model, msa_onehot)
        end
        
        # Reshape to (nseqs, nhidden)
        nhidden = prod(size(hidden_cpu)[1:end-1])
        hidden_reshaped = reshape(hidden_cpu, nhidden, nseqs)'
        
        return hidden_reshaped
    end
    
    # Process in batches
    println("Processing $(nseqs) sequences in batches of $(batch_size)...")
    nbatches = ceil(Int, nseqs / batch_size)
    
    # Move model to GPU once if using GPU
    rbm_device = use_gpu && CUDA.functional() ? gpu(rbm_model) : rbm_model
    
    hidden_batches = []
    
    @showprogress for i in 1:nbatches
        start_idx = (i - 1) * batch_size + 1
        end_idx = min(i * batch_size, nseqs)
        
        # Get batch sequences
        batch_sequences = sequences[start_idx:end_idx]
        
        # Encode batch to one-hot
        batch_onehot = encode_sequences_to_onehot(batch_sequences, tokens)
        
        # Compute hidden activations
        if use_gpu && CUDA.functional()
            batch_gpu = gpu(batch_onehot)
            hidden_batch = sample_h_from_v(rbm_device, batch_gpu)
            hidden_batch_cpu = cpu(hidden_batch)
        else
            hidden_batch_cpu = sample_h_from_v(rbm_device, batch_onehot)
        end
        
        push!(hidden_batches, hidden_batch_cpu)
    end
    
    # Concatenate all batches
    println("Concatenating batches...")
    hidden_all = cat(hidden_batches..., dims=ndims(hidden_batches[1]))
    
    # Reshape to (nseqs, nhidden)
    nhidden = prod(size(hidden_all)[1:end-1])
    hidden_reshaped = reshape(hidden_all, nhidden, nseqs)'
    
    return hidden_reshaped
end

function main()
    args = parse_commandline()
    # Check input files exist
    if !isfile(args["input"])
        error("Input file $(args["input"]) does not exist.")
    end
    if !isfile(args["rbm_model_path"])
        error("RBM model file $(args["rbm_model_path"]) does not exist.")
    end
    
    # Setup device
    use_gpu = args["use_gpu"] && CUDA.functional()
    if use_gpu
        println("Using GPU for encoding")
    else
        println("Using CPU for encoding")
    end
    
    # Define protein tokens (21 tokens: 20 amino acids + gap)
    tokens = collect("-ACDEFGHIKLMNPQRSTVWY")
    println("Using $(length(tokens)) tokens for encoding")
    
    # Load RBM model
    println("Loading RBM model from $(args["rbm_model_path"])...")
    rbm_model = load_rbm(args["rbm_model_path"])
    println("RBM model loaded successfully")
    
    # Load CSV file
    println("Loading input dataset from $(args["input"])...")
    df = CSV.read(args["input"], DataFrame)
    
    # Check required columns
    if !(args["column_sequences"] in names(df))
        error("Column '$(args["column_sequences"])' not found in CSV file")
    end
    if !(args["column_headers"] in names(df))
        error("Column '$(args["column_headers"])' not found in CSV file")
    end
    
    sequences = String.(df[!, args["column_sequences"]])
    headers = String.(df[!, args["column_headers"]])
    labels = args["column_labels"] in names(df) ? Vector(df[!, args["column_labels"]]) : nothing
    
    println("Loaded $(length(sequences)) sequences from CSV file")
    if !isnothing(labels)
        println("Found labels with $(length(unique(labels))) unique values")
    end
    
    # RBM encode sequences
    println("RBM encoding aligned sequences...")
    rbm_embeddings = rbm_encode_sequences(sequences, rbm_model, tokens, use_gpu, args["batch_size"])
    println("RBM encoded shape: $(size(rbm_embeddings))")
    
    # Prepare output filename
    output_prefix = splitext(args["input"])[1]
    output_path = "$(output_prefix).rbm_drelu.npz"
    
    # Save to npz file using Python's numpy via PyCall
    println("Saving RBM encoding to $(output_path)...")
    np = pyimport("numpy")
    
    if !isnothing(labels)
        # Convert labels to proper type if needed
        labels_array = collect(labels)
        np.savez_compressed(output_path,
            embeddings=rbm_embeddings,
            labels=labels_array,
            headers=headers
        )
    else
        np.savez_compressed(output_path,
            embeddings=rbm_embeddings,
            headers=headers
        )
    end
    
    println("Successfully saved RBM encoding to $(output_path)")
    println("Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
