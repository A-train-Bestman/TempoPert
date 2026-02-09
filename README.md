# TempoPert: Deep learning of temporal gene expression trajectories enables time-resolved drug response prediction and repositioning

This repository hosts the official implementation of TempoPert, a deep learning framework designed to model the temporal evolution dynamics of drug perturbations. By simulating the progression from basal state through early and late response phases, TempoPert captures the temporal trajectories of drug effects that static models miss.

![TempoPert](img/TempoPert.jpg)

## Highlights

- **Time-Resolved Modeling**: Predicts gene expression at 6h and 24h timepoints, capturing both early and late drug responses
- **Superior Performance**: Achieves PCC of 0.8649 (compound-split), outperforming state-of-the-art baselines
- **Trajectory-Guided Learning**: Employs biologically-constrained trajectory optimization for accurate temporal dynamics
- **Diverse Applications**: Enables drug sensitivity prediction, drug repositioning with cell-type-specific therapeutic windows, and counterfactual simulation

## Download Model and Datasets

We provide preprocessed datasets (LINCS L1000) and model checkpoints for training and reproducibility.

### Download Links

**Datasets**:

- [LINCS L1000 with splits](https://github.com/A-train-Bestman/TempoPert/tree/main/dataset) - `Lincs_L1000.h5ad`
- [Time series data](https://github.com/A-train-Bestman/TempoPert/tree/main/dataset) - `L1000_0_6_24.csv`
- [KPGT molecular embeddings](https://github.com/A-train-Bestman/TempoPert/tree/main/dataset) - `KPGT_2304.pkl`

### Data Sources

The L1000 dataset is from the LINCS project (GSE92742). Alternative download:

- LINCS official website: https://maayanlab.cloud/sigcom-lincs/
- GEO database: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742

To clone our repository:

```bash
git clone https://github.com/A-train-Bestman/TempoPert.git
```

Please download the datasets and store them in the `dataset` folder. Download the trained weights from [checkpoint scouse](https://drive.google.com/drive/folders/1nEsu-0hLoHPSPDgxqCLQ1I9Y2hEEQ0-c).

## Repository Structure

```
TempoPert/
├── model/: contains the model architecture and training code
│   ├── bulider.py: trajectory builder for preprocessing
│   ├── progressive_model.py: TempoPert model architecture
│   ├── train_TempoPert.py: training script
│   └── trajectory_dataset.py: data loader utilities
│   └── train_memory.py: training trajectory builder script
├── dataset/: contains datasets (download separately)
├── checkpoint/: model checkpoints
├── trainer/: baseline model trainer
```

## Step 1: Installation

We recommend using Anaconda to create a conda environment:

```bash
conda create -n tempopert python=3.8
conda activate tempopert
```

Install PyTorch (adjust according to your CUDA version):

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

Install other dependencies:
### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (for GPU acceleration)
- scanpy, anndata, pandas, numpy, scipy, scikit-learn
- rdkit, tqdm, matplotlib

## Step 2: Preprocess Trajectory Data

Before training, you need to build optimized temporal trajectories (one-time process):

```bash
cd model
python train_memory.py \
    --adata-path dataset/Lincs_L1000_with_pairs_splits.h5ad \
    --timeseries-path dataset/L1000_0_6_24.csv \
    --memory-dir ./trajectory_memory \
    --optimize-steps 30 \
    --only-optimize
```

This generates preprocessed trajectory data in `./trajectory_memory/`.

## Step 3: Train with Provided Dataset

### Train on L1000 Dataset

**Compound Split** (evaluate generalization to new compounds):

```bash
python train_TempoPert.py --adata-path ../dataset/Lincs_L1000_with_pairs_splits.h5ad \
                          --timeseries-path ../dataset/L1000_0_6_24.csv \
                          --split-key drug_splits_4 \
                          --batch-size 2048 \
                          --epochs 100 \
                          --device cuda
```

**Cell Line Split** (evaluate generalization to new cell types):

```bash
python train_TempoPert.py --split-key cell_splits_4 \
                          --batch-size 2048 \
                          --epochs 100
```

**Random Split** (performance upper bound):

```bash
python train_TempoPert.py --split-key random_splits_4 \
                          --batch-size 2048 \
                          --epochs 100
```
