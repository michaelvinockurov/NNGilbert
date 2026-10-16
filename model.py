import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict
import random
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from config import ALLELE_VOCAB, NUM_ALLELE_CLASSES


class ConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualHeadCNN(nn.Module):


    def __init__(
        self,
        input_channels: int = 11, 
        input_length: int = 46,
        num_classes: int = NUM_ALLELE_CLASSES,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        self.num_classes = num_classes

     
        self.features = nn.Sequential(
            ConvBlock(input_channels, 64, kernel_size=7, dropout=dropout),
            ConvBlock(64, 64, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(2),   
            ConvBlock(64, 128, kernel_size=5, dropout=dropout),
            ConvBlock(128, 128, kernel_size=3, dropout=dropout),
            nn.MaxPool1d(2),  
            ConvBlock(128, 256, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),  
        )

        
        self.feature_dim = 256

        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

      
        self.head1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        
        self.head2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

       
        self._init_weights()

    def _init_weights(self):
       
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       
        features = self.features(x).squeeze(-1) 
        shared = self.shared_fc(features)
        logits1 = self.head1(shared)
        logits2 = self.head2(shared)
        return logits1, logits2

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
     
        logits1, logits2 = self.forward(x)
        pred1 = logits1.argmax(dim=1)
        pred2 = logits2.argmax(dim=1)
        return pred1, pred2


class FocalLoss(nn.Module):
    

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,  
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
       
        n_classes = inputs.size(1)
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = torch.log_softmax(inputs, dim=1)
        probs = torch.softmax(inputs, dim=1)

        
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

       
        focal_weight = (1 - pt) ** self.gamma

        
        if self.label_smoothing > 0:
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            ce_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        
        if self.weight is not None:
            class_weights = self.weight[targets]
            ce_loss = ce_loss * class_weights

        loss = focal_weight * ce_loss
        return loss.mean()


class DualHeadLoss(nn.Module):
  
    def __init__(
        self,
        weights1: Optional[torch.Tensor] = None,
        weights2: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05, 
        gamma: float = 1.5,
        use_focal: bool = False,  
    ):
        super().__init__()
        if use_focal:
            self.criterion1 = FocalLoss(
                weight=weights1, gamma=gamma, label_smoothing=label_smoothing
            )
            self.criterion2 = FocalLoss(
                weight=weights2, gamma=gamma, label_smoothing=label_smoothing
            )
        else:
            self.criterion1 = nn.CrossEntropyLoss(
                weight=weights1, label_smoothing=label_smoothing
            )
            self.criterion2 = nn.CrossEntropyLoss(
                weight=weights2, label_smoothing=label_smoothing
            )

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        targets1: torch.Tensor,
        targets2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss1 = self.criterion1(logits1, targets1)
        loss2 = self.criterion2(logits2, targets2)
        total_loss = loss1 + loss2
        return total_loss, loss1, loss2
