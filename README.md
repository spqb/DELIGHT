# DELIGHT

> [!WARNING]
> This repository is still under development. We will publish the code soon!

DELIGHT (Data-Enritched Label-Informed Generation of Homologous sequences using Transformers) is a convoluted acronym that attempts to condense the content of the paper _"Data augmentation enables label-specific generation of homologous protein sequences"_.

This repository contains the code for reproducing the pipeline described in the paper and the jupyter notebooks that allow to generate the figures.

## ⬇️ Installation
To be able to use the code, you need to install some dependencies. In particular, you need

- The [transformes](https://huggingface.co/docs/transformers/installation) library by 🤗 HuggingFace
- Some utilities from the package [adabmDCA](https://github.com/spqb/adabmDCApy.git)
- The general purpose [rbms](https://github.com/DsysDML/rbms.git) package for training the encoding RBM
- The package [annaDCA](https://github.com/rossetl/annaDCA.git) for the label-assisted RBM generative model

We recommand creating a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment with `python >= 3.11` and installing the dependencies inside it:

```bash
conda create --n delight python=3.12
conda activate delight
python -m pip install -r requirements.txt
```

## 💾 Data availability
The data required for reproducing the paper's experiments can be found in the [Zenodo repository](https://zenodo.org/records/15979182).

## 🔁 Reproducing the results
First off, download the Zenodo repository `datasets` and enter the `DELIGHT` folder. Create the directory `experiments` and move the `datasets` repository inside it:

```bash
cd DELIGHT
mkdir experiments
mv /path/to/datasets experiments/
```

### ✳️ Construct the embeddings and infer specificities
For each dataset, we are going to create the train and test data embeddings using the __ProtBERT__ pLM. We consider both the zero-shot embedding using the __foundation__ model and also the embedding obtained after fine-tuning ProtBERT on the training data using the __contrastive__ objective.


### 🎲 Label-aware homologous sequences generation


