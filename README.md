# Decoding-Memory-Loss-in-Alzheimer-s-Disease-Through-Machine-Learning-and-Multi-Omics-Integration

## Overview

This repository contains the R and Python scripts used to investigate Alzheimer's disease (AD)-associated gene expression changes involving a memory-annotated candidate gene set. The workflow combines differential gene expression analysis, single-nucleus RNA sequencing (snRNA-seq) analysis, and machine-learning classification to examine transcriptional changes associated with AD.

The repository includes scripts for differential expression analysis, cellular-level investigation, and classification of AD and control samples using Random Forest (RF), XGBoost (XGB), and a multilayer perceptron (MLP).

## Repository Contents

| File | Description |
|---|---|
| `DEG1.R` | R script for differential gene expression analysis. |
| `DEG2.R` | R script for batch corrected differential gene expression analyses. |
| `snRNA-seq.R` | R script for single-nucleus RNA sequencing data processing and downstream analysis. |
| `RF_ADvsCTL_model.py` | Python script implementing a Random Forest classifier to distinguish AD from control samples. |
| `XGB_ADvsCTL.py` | Python script implementing an XGBoost classifier for AD-versus-control classification. |
| `MLP_ADvsCTL_model.py` | Python script implementing a multilayer perceptron classifier for AD-versus-control classification. |
| `memory genes.xlsx` | Spreadsheet containing the memory-annotated candidate gene list used in the study. |

## Data Availability

The scripts are intended for use with the gene-expression datasets specified in the manuscript. Users should consult the associated study documentation for dataset accession numbers, sample information, and preprocessing details.

The `memory genes.xlsx` file provides the candidate gene list used in the analysis.

## Reproducibility

For reproducible results:

- Use the same dataset versions and preprocessing procedures described in the associated study.
- Record the R and Python versions and package dependencies.
- Keep track of model parameters and dataset partitions where applicable.

## Citation

If you use this repository in your research, please cite the associated publication once available. Until then, cite the repository URL and its release or commit identifier where appropriate.

## License

See the `LICENSE` file for the applicable terms of use.
