**UGT1A1 Genotyping using 1D-CNN Ensemble**

_Official PyTorch implementation for the automated classification of UGT1A1 rs3064744 polymorphism using asymmetric PCR melting curve analysis (aPCR-MCA)._

This repository contains the deep learning pipeline used to achieve 100% cross-platform accuracy in discriminating highly homologous alleles (including the challenging 7TA and 8TA variants) directly from raw fluorescence melting curves.

**Overview**

Genotyping the highly AT-rich UGT1A1 rs3064744 promoter region is critical for preemptive pharmacogenetic testing (e.g., preventing irinotecan toxicity) and diagnosing Gilbert’s syndrome.

Instead of relying on error-prone manual interpretation of melting peaks, this project introduces an ensemble of one-dimensional convolutional neural networks (1D-CNN).
The core innovation is the Dual Control Sample Comparison mechanism: the network processes an 11-channel tensor that includes the raw signal, its derivatives, and the thermodynamic differences between the patient's sample and known reference anchors (7TA/7TA and 7TA/8TA). This forces the AI to focus on genuine biological features rather than instrument-specific thermal noise.

**Key Features:**
End-to-end Pipeline: From raw MCA fluorescence data to a predicted genotype with a confidence score.

Dual-Head Architecture: Separate prediction heads for Allele 1 and Allele 2 to capture all 10 possible genotype combinations in a single tube.

Resilient to Noise: Uses Test-Time Augmentation (TTA) and ensemble averaging for highly robust clinical predictions.



**Repository Structure**

model.py — The primary script containing the full 1D-CNN architecture, data loaders, training pipeline (with mixup augmentation, early stopping, and OneCycleLR scheduler), and evaluation metrics.

infer.py — A stripped-down, highly optimized script strictly for running inference on new data using pre-trained models.

**How to Use**

1. Data Format

The scripts expect a Tab-Separated Values (.tsv) file. Each row represents a single PCR reaction well.

Columns 1–45: Raw fluorescence measurements across the temperature gradient.

Control Flag Column: An integer indicating the sample type:

0 = Unknown clinical sample (to be predicted)

1 = 7TA/8TA Reference Anchor

2 = 7TA/7TA Reference Anchor

(Note: Every inference run must include at least one 7_8 and one 7_7 control in the dataset to build the relative differential channels).

2. Training the Ensemble

To train a new ensemble of 5 models from scratch using your dataset:

python model.py train --data dataset.tsv --output models/ --ensemble 5 --epochs 200


3. Running Inference

To classify new patient samples using a trained ensemble, use the lightweight inference script:

python infer.py --data new_patients.tsv --model models/ --output results.tsv


**Output Example:**

Row   1: 6_7    (conf: 0.99, 0.98)

Row   2: 7_7    (conf: 1.00, 1.00)

Row   3: 7_8    (conf: 0.94, 0.96) ⚠️

Row   4: 6_6    (conf: 0.99, 0.99)


The script automatically flags predictions with lower confidence thresholds for manual review.

