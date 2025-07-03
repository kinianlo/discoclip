# DisCoCLIP

This repository contains the code for the paper "DisCoCLIP" which focuses on replacing the text encoder in CLIP with Distributional Compositional (DisCo) Models.

# Datasets
We train and test our model on [SVO-Probes](https://github.com/google-deepmind/svo_probes) and [ARO Benchmark](https://github.com/mertyg/vision-language-models-are-bows).

# General pipeline for CCG models

A sentence is first parsed into diagrams using a cached parser. Then the diagrams are converted into tensor networks using the `MPSAnsatz` with a chosen bond dimension. 
Before we do any training, we need to build up a large set of symbols, which will each be assigned an integer ID. 
We do this because we want to follow the same Pytorch convention, which there is a preprocessor that converts a sentence as a string into a list of integers. The only difference is that for each network, we also have to specify the einsum string that describes how the tensors are contracted.

# General pipeline for non-CCG models

For non-CCG models, we will have another preprocessor that converts a sentence into a list of integers, but there is no need to specify the einsum string, as the tensors are contracted in a fixed way.

# Models

We have implemented the following models:
- `Compact`: A CCG model that uses tensors of different ranks for different types of words.
- `Tree`: A CCG model that represents each word as a vector and uses rank-3 tensors to compose them. 
- `Cups`: A non-CCG model that uses a matrix for each word and neighbouring words are contracted in a fixed way.
- `Spider`: Non-CCG models that use element-wise addition and multiplication, respectively, to compose the words.

# Preparation

Before training, some preparation is needed. The following steps should be followed:

1. Install `discoclip`, which can be done by cloning this repository and install it using pip:
```bash
git clone https://github.com/kinianlo/discoclip.git
cd discoclip
pip install .
```

2. Download the zip file which contain all the images in the ARO Benchmark from [this Google Drive link](https://drive.google.com/uc?export=download&id=1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9) and move it to the `data/raw/aro` directory. Decompress the zip file and rename the folder to `images`. To ensure the images are in the correct directory, check if `data/raw/aro/images/21.jpg` exists. 


3. Precompute the embeddings for the images in the ARO Benchmark. This can be done by running the following command:
```bash
python scripts/encode_aro_images.py
```
<!-- By default, the CPU is used, but you can use a GPU by specifying `--device cuda`. -->
By default, the CPU is used to host the clip model, but you can specify the device to use by adding `--device cuda` to the command if you have a GPU available. Use `--device mps` if you are on a Mac with Apple Silicon.

# Training (ARO Benchmark)
To train a model for the ARO dataset, you can use the following command:
```bash
python discoclip/train_aro.py --config configs/aro_default.yaml
```
To see the available options, you can run:
```bash
python discoclip/train_aro.py --help
```

### Training options
You can modify the training options in the `configs/aro_default.yaml` file. The following is a table of the available options:

| Group | Argument | Description | Default Value |
|---|---|---|---|
| Data | `--train-data-path` | Path to the training data JSON file | `data/processed/aro/combined/train.json` |
| Data | `--val-data-path` | Path to the validation data JSON file | `data/processed/aro/combined/val.json` |
| Data | `--test-data-path` | Path to the test data JSON file | `data/processed/aro/combined/test.json` |
| Data | `--image-lookup-path` | Path to the image lookup file | `data/processed/aro/clip_ViT-B-32.pt` |
| Model | `--reader` | Type of reader to use for text processing | `bobcat` |
| Model | `--embedding-dim` | Dimension of the embeddings | `512` |
| Model | `--bond-dim` | Bond dimension for the MPS ansatz | `10` |
| Training | `--batch-size` | Batch size for training | `64` |
| Training | `--learning-rate` | Learning rate for the optimizer | `0.003` |
| Training | `--weight-decay` | Weight decay for the optimizer | `0.01` |
| Training | `--epochs` | Number of epochs to train the model | `0` |
| Training | `--patience` | Patience for early stopping | `5` |
| Loss | `--temperature` | Temperature for the InfoNCE loss | `0.07` |
| Loss | `--hard-neg-loss-weight` | Weight for the hard negative loss component | `0` |
| Loss | `--hard-neg-margin` | Margin for the hard negative loss | `0.1` |
| Loss | `--hard-neg-distance-function` | Distance function for the hard negative loss | `cosine` |
| Loss | `--hard-neg-swap` | Whether to use the distance swap for the hard negative loss | `false` |
| Logging | `--log-path` | Path to save training logs | `logs` |
| Logging | `--checkpoint-path` | Path to save model checkpoints | `checkpoints` |
| Logging | `--mlflow-experiment` | Name of the MLflow experiment | `discoclip_aro` |
| System | `--device` | Device to run the training on | `cpu` |
| System | `--seed` | Random seed for reproducibility | `42` |
