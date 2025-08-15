import torch
import numpy as np

def calculate_iou(predictions, ground_truth, threshold=0.5):
    """Calculate Intersection over Union for binary segmentation"""
    predictions = (predictions > threshold).float()
    mask = (ground_truth == 1).float()
    
    intersection = (predictions * mask).sum(dim=(1, 2))
    union = (predictions + mask).sum(dim=(1, 2)) - intersection
    return (intersection / (union + 1e-6)).mean()

def calculate_precision_recall_f1(preds, labels):
    """Calculate precision, recall and F1-score"""
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return float(precision), float(recall), float(f1_score)