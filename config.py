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

RANDOM_SEED = 42
VALIDATION_SPLIT = 0.20  
BATCH_SIZE = 32
LEARNING_RATE = 8e-4  
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 40


ALLELE_VOCAB = ["5", "6", "7", "8", "K-"]
ALLELE_TO_IDX = {a: i for i, a in enumerate(ALLELE_VOCAB)}
IDX_TO_ALLELE = {i: a for a, i in ALLELE_TO_IDX.items()}
NUM_ALLELE_CLASSES = len(ALLELE_VOCAB)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed: int = RANDOM_SEED):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
