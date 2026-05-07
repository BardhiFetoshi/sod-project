# Salient Object Detection using CNN

This project implements a Salient Object Detection (SOD) model using a custom CNN in PyTorch. The model takes an image as input and outputs a binary mask highlighting the most important object.

## Dataset

The model was trained and evaluated on the ECSSD dataset, using training, validation, and test splits.

## Model

A CNN-based encoder–decoder architecture (SODCNN) is used to generate saliency masks from input images.

## Results

IoU: 0.4315  
Precision: 0.5455  
Recall: 0.7431  
F1-score: 0.5761  
MAE: 0.2111  

The model detects most salient regions well, but sometimes includes extra background areas.

## How to Run

Install dependencies:

pip install -r requirements.txt

Open and run:

sod_demo.ipynb

## Files

- src/ → model and utilities  
- sod_demo.ipynb → demo notebook  
- requirements.txt → dependencies  
- SOD_Project_Report_FF.pdf → report  

## Conclusion

This project demonstrates a complete SOD pipeline using a CNN model in PyTorch.
