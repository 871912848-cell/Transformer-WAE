# Transformer-WAE

This repository provides the implementation of a Transformer-based Wasserstein autoencoder (Transformer-WAE) for spectral augmentation and starch content prediction using hyperspectral spectra of single wheat kernels.

The model jointly learns hyperspectral spectra and starch content labels, and generates synthetic spectrum–starch pairs for calibration-set augmentation. The code also includes training metrics, generated-data evaluation, attention heatmap export, and synthetic sample generation.

## Overview

The main functions of this repository include:

- Loading hyperspectral spectra and starch content from Excel files
- Normalizing spectra and starch labels for joint modeling
- Training a Transformer-WAE model using reconstruction loss and MMD regularization
- Generating synthetic spectrum–starch pairs
- Evaluating generated spectra using MSE, SAM, MMD, SVD-based distance, and PRD metrics
- Exporting encoder or decoder attention matrices and heatmaps
- Saving generated spectra and corresponding starch labels to Excel files

## Model Structure

The Transformer-WAE framework consists of:

1. **Encoder**  
   The encoder maps the input spectrum–label vector into a latent representation.

2. **Decoder**  
   The decoder reconstructs the input and generates new synthetic spectrum–label pairs from random latent vectors.

3. **MMD regularization**  
   Maximum mean discrepancy is used to match the encoded latent distribution with a prior distribution.

4. **Attention visualization**  
   Multi-head self-attention weights can be exported from the encoder or decoder to analyze wavelength-level relationships.

## File Description

```text
Transformer-WAE/
│
├── transformer_wae.py        # Main implementation of Transformer-WAE
├── README.md                 # Project description
├── requirements.txt          # Required Python packages
└── output/                   # Generated results, figures, and Excel files
```

## Requirements

The code was developed using Python and PyTorch. The main required packages include:

```text
numpy
pandas
matplotlib
torch
openpyxl
```

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Input Data Format

The input file should be an Excel file in which:

- Each row represents one wheat kernel sample.
- All columns except the last column are spectral variables.
- The last column is the starch content value.

Example:

```text
band_1 | band_2 | band_3 | ... | band_n | starch
```

## Usage

Modify the input file path in the main function:

```python
excel_path = r"your_data_path.xlsx"
```

Then run:

```bash
python transformer_wae.py
```

After training, the program will automatically save:

- Training metrics in `.csv` format
- Loss and distribution metric curves
- Attention matrices in `.xlsx` format
- Attention heatmaps in `.png` format
- Generated spectrum–starch pairs in `.xlsx` format

## Output Files

The main output files include:

```text
training_metrics.csv
recon_over_epochs.png
mmdz_over_epochs.png
total_over_epochs.png
mse_over_epochs.png
sam_over_epochs.png
mmd_spec_over_epochs.png
svd_over_epochs.png
prd_f1_over_epochs.png
generated_pairs.xlsx
decoder_layer*_attn.xlsx
decoder_layer*_head*.png
decoder_layer*_mean_heads.png
```

## Notes

The current implementation uses fixed evaluation samples and fixed latent vectors during training evaluation, which helps make the generated-data metrics more stable and comparable across epochs.

The generated spectra are intended for calibration-set augmentation and should be further evaluated before use in downstream regression models.

## Data Availability

The full hyperspectral dataset is not included in this repository. It may be available from the corresponding author upon reasonable request.

## Citation

If this code is useful for your research, please cite the related manuscript:

```text
Starch-associated spectral signatures in single wheat kernels revealed by dual-range hyperspectral imaging and Transformer-WAE augmentation.
```

## License

This repository is released for academic and research use.
