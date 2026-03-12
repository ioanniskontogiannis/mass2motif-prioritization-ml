# Prioritizing Mass2Motifs for Metabolite Annotation using Supervised Machine Learning



## Overview
This repository contains all notebooks and scripts used to perform data preprocessing, MS2LDA analysis, feature extraction, and machine‑learning–based prioritization of Mass2Motifs.
The workflow was developed as part of the MSc thesis Prioritizing Mass2Motifs for Metabolite Annotation using Supervised Machine Learning.


## Repository Structure

```
├── Spec2Vec_retraining
│   ├── _negative_ref_spec_library.py
│   ├── retrain_s2v_neg.py
│   ├── validate_s2v.ipynb
│   └── validate_s2v_bigger_scale.ipynb
├── chemical_class
│   ├── chemical_class_neg_ChEBI.ipynb
│   ├── chemical_class_pos_ChEBI.ipynb
│   └── chemical_diversity_with_COCONUT.ipynb
├── filtering_and_deduplication
│   ├── filtering_Spec2Vec_retraining.ipynb
│   └── library_filtering_MS2LDA_input.ipynb
├── machine_learning
│   ├── ML_neg_external_validation.ipynb
│   ├── ML_pos_external_validation.ipynb
│   ├── __pycache__
│   │   └── motif_utils.cpython-312.pyc
│   ├── external_validation_plots.ipynb
│   ├── modeling_internal_validation.ipynb
│   ├── motif_priority_model.pkl
│   ├── motif_priority_model_2.pkl
│   ├── motif_utils.py
│   ├── prioritization_workflow.py
│   └── testing_ml_models.ipynb
├── motif_curation
│   ├── Mass2Motif_filtering.ipynb
│   ├── processed_reproducible_motifs.ipynb
│   ├── reproducibility_check.ipynb
│   └── stable_motifs_curation.ipynb
└── ms2lda_runs
    ├── default_run.py
    ├── ms2lda_neg.py
    ├── ms2lda_run.py
    ├── ms2lda_server_run.py
    └── test_run.py
```

## Folder descriptions

- **Spec2Vec_retraining** — Scripts and notebooks for retraining Spec2Vec models, including negative‑mode reference library generation and validation workflows.
- **chemical_class** — Notebooks for chemical investigation of MSnLib using ChEBI and NPClassifier's predictions of the COCONUT database.
- **filtering_and_deduplication** — Preprocessing notebooks for filtering and deduplication of the dataset that was used as the reference library and retraining of Spec2Vec, and also deduplication of the MSnLib that was later used as input for MS2LDA
- **machine_learning** — All machine‑learning components: internal and external validation notebooks, trained models (`.pkl`), utility functions (`motif_utils.py`), and the full prioritization workflow (`prioritization_workflow.py`).
- **motif_curation** — Notebooks for Mass2Motif curation, reproducibility checks, and generating stable motif sets used in downstream modeling.
- **ms2lda_runs** — Scripts for running MS2LDA in different configurations (default, negative mode, server‑based runs, and test runs for external validation).


## Installation

This project uses the same software environment as **MS2LDA**, since several dependencies (e.g., matchms, gensim, numpy, scipy) must match the versions used during MS2LDA runs. To ensure compatibility, install and activate the MS2LDA environment before running any notebooks or scripts in this repository.

### 1. Install the MS2LDA environment

```bash
git clone https://github.com/vdhooftcompmet/MS2LDA.git
cd MS2LDA
conda env create -f MS2LDA_environment.yml
conda activate MS2LDA_v2
./run_analysis.sh --only-download
```
### 2. Clone this repository

```bash
git clone https://github.com/ioanniskontogiannis/mass2motif-prioritization-ml.git
cd mass2motif-prioritization-ml
```
## Usage

The full Mass2Motif prioritization workflow is executed through the script  
`machine_learning/prioritization_workflow.py`. This script performs all required steps:

- loads the MS2LDA output  
- runs additional Mass2Motif filtering  
- applies the trained machine‑learning model  
- generates a prioritized list of Mass2Motifs  
- saves the final Excel file (`put_a_name.xlsx`) in your designated output folder  

### 1. Navigate to the scripts/machine_learning folder

```bash
cd scripts/machine_learning
```
### 2. Edit `prioritization_workflow.py`

Open the script and modify the two paths one at the top and one the bottom:

- `ms2lda_output_folder` — the folder containing your MS2LDA results  
- `output_path` — where the prioritized Excel file should be saved  
  (usually the same MS2LDA output folder)

Example:

```python
RUN_PATH = "/path/to/your/MS2LDA/output/"
output_folder = "/path/to/your/MS2LDA/output/"
```
### 3. Run the workflow

Make sure the MS2LDA environment is activated, then run:

```bash
python prioritization_workflow.py
```
### 4. Output

The output is an Excel file with the prioritized Mass2Motifs. The name and the output folder has been set previously in step 2


## Data Availability

All spectral libraries, MS2LDA outputs, machine‑learning training data, the trained prioritization model, external validation results, and all reproducible Mass2Motifs used and produced in this study are available in Zenodo: https://doi.org/10.5281/zenodo.18968803

Every input and output of the study is included, together with information and instructions on how to fully reproduce the workflow.

The new Spec2Vec model for negative ionization mode, along with all required supporting files (reference library, embeddings), is available in the MS2LDA repository:

https://github.com/vdhooftcompmet/MS2LDA

## Citation

If you use this workflow, code, or data, please cite:

Kontogiannis, I., Torres-Ortega, L. R., & van der Hooft, J. (2026). *Prioritizing Mass2Motifs for Metabolite Annotation using Supervised Machine Learning* (Version v1.0). Zenodo. https://doi.org/10.5281/zenodo.18968803

## Contact

For questions, feedback, or collaboration, feel free to reach out:

**Ioannis Kontogiannis**  
Wageningen University & Research  
Email: **ioannis.kontogiannis@wur.nl**

