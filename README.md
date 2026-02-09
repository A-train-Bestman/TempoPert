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
