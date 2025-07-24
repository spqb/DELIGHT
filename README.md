# DELIGHT

**DELIGHT** (Data-Enriched Label-Informed Generation of Homologous sequences using Transformers) is a slightly convoluted acronym that condenses the content of the paper:  
_"Data augmentation enables label-specific generation of homologous protein sequences"_.

This repository provides:
- The code to reproduce the full pipeline described in the paper.
- Jupyter notebooks to regenerate the figures used in the publication.

---

## ⬇️ Installation

To use this code, you’ll need to install several dependencies:

- [`transformers`](https://huggingface.co/docs/transformers/installation) by 🤗 HuggingFace  
- Utilities from [`adabmDCA`](https://github.com/spqb/adabmDCApy.git)  
- The general-purpose [`rbms`](https://github.com/DsysDML/rbms.git) package for training RBMs  
- Our custom [`annaDCA`](https://github.com/rossetl/annaDCA.git) package for label-informed RBM generation  

We recommend using a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment with Python ≥ 3.11:

```bash
conda create -n delight python=3.12
conda activate delight
python -m pip install -r requirements.txt
```

> [!WARNING]
> Due to a temporary incompatibility between the package `rbms` and `torch==2.6`, you need to follow the following procedure to install `rbms`. First, install the package through the GitHub repository. Move to a base repository and then do
> ```bash
> git clone https://github.com/DsysDML/rbms.git
> cd rbms
>```
> Then, open the file `pyproject.toml` and, under the flag `dependencies`, do the change: `"torch>=2.0.0, <=2.5.0"` $\rightarrow$ `"torch>=2.0.0, <=2.6.0"`.
> After that, you can manually install the package inside the conda environment you just created
> ```bash
> python3 -m pip install .
> ```

---

## 🌞 Quickstart

### 📊 Input Data Format

The pipeline requires two input files:
- A **training file** with annotated sequences
- A **query file** with sequences to be annotated

**Training file** (CSV format) must include:
- `header`: sequence identifiers  
- `sequence`: full-length sequences  
- `sequence_align`: aligned versions of the sequences  
- `label`: functional or structural annotations  

> Custom column names can be provided via CLI arguments.

**Query file** can be either:
- A FASTA file  
- A CSV with at least `header` and `sequence` columns

---

### 🔎 Embedding & Annotation Prediction

To embed the query sequences using a protein Language Model (pLM) and predict their annotations, run:

```bash
python3 ./src/pLM_encoding.py \
    --train <training_file> \
    --query <query_file> \
    --flag <flag_name> \
    --zero-shot \
    --bf16
```

- `<flag_name>` is a string added to output files for traceability.
- `--zero-shot` uses the foundation model without fine-tuning.

To see all available options:

```bash
python3 ./src/pLM_encoding.py -h
```

#### ⚙️ Optional CLI Arguments
- `--column_headers`: defaults to `header`
- `--column_sequences`: defaults to `sequence`
- `--column_labels`: defaults to `label`

#### 📤 Output
- `.npz` file with train embeddings  
- `.npz` file with query embeddings, predicted labels, and confidence scores  
- `.csv` file with query sequences and predicted labels  

---

### 🧠 Training an Annotation-Assisted RBM

Once predictions are available, you can train a label-aware RBM model:

```bash
annadca train \
    -d <data_file.csv> \
    -o <output_dir> \
    --column_names <column_headers> \
    --column_sequences <column_sequences_align> \
    --column_labels <column_labels> \
    --nepochs 30000 \
    --nchains 5000
```

- `data_file.csv` is the output from the previous embedding step.
- The model will be saved in `<output_dir>`.

> [!NOTE]  
> `<column_sequences_align>` **must contain aligned sequences**. Full-length sequences are not accepted.

> [!NOTE]  
> For better performance, we recommend **merging the CSV** file containing predicted labels with the original training file used in `pLM_encoding.py`.

---

### 🎯 Conditional Sequence Generation

To generate new sequences based on specific annotations using the trained RBM model, refer to:  
`./notebooks/Conditioned_generation.ipynb`

---

## 🔁 Reproducing the Paper Results

To replicate the results from the paper:

1. Create the necessary directories and download the datasets:

```bash
cd DELIGHT
mkdir experiments && cd experiments
wget https://zenodo.org/records/15979182/files/datasets.zip
unzip datasets.zip && rm datasets.zip
mkdir models && cd ..
```

2. Run the script to train the models and compute embeddings:

```bash
chmod +x ./bash/reproduce_paper_results.sh
./bash/reproduce_paper_results.sh
```

> [!WARNING]  
> This step is computationally intensive and may take a while.

---

## 📈 Reproducing the Figures

- **Section II-B (_Data augmentation_)**:  
  Use `./notebooks/Classification.ipynb`

- **Section II-C (_Label-specific generation_)**:  
  Use `./notebooks/Conditioned_generation.ipynb` and `./notebooks/False_positives_analysis.ipynb`

- **Additional visualizations**:  
  Use `./notebooks/Additional_figures.ipynb`

---

## 📚 Citing
```
@misc{rosset2025dataaugmentationenableslabelspecific,
      title={Data augmentation enables label-specific generation of homologous protein sequences}, 
      author={Lorenzo Rosset and Martin Weigt and Francesco Zamponi},
      year={2025},
      eprint={2507.15651},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2507.15651}, 
}
```