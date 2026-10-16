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
from data import parse_genotype
from config import ALLELE_VOCAB, NUM_ALLELE_CLASSES, set_seed


class ModelTrainer:
    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler, criterion, device
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.best_val_loss = float("inf")

    def mixup_data(
        self, x: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, alpha: float = 0.2
    ):

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y1_a, y1_b = y1, y1[index]
        y2_a, y2_b = y2, y2[index]

        return mixed_x, y1_a, y1_b, y2_a, y2_b, lam

    def mixup_criterion(
        self,
        criterion: nn.Module,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def compute_class_weights(
        self, genotypes: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        counts1 = {a: 0 for a in ALLELE_VOCAB}
        counts2 = {a: 0 for a in ALLELE_VOCAB}

        for g in genotypes:
            a1, a2 = parse_genotype(g)
            counts1[a1] += 1
            counts2[a2] += 1

        def make_weights(counts):
            total = sum(counts.values())
            weights = []
            for allele in ALLELE_VOCAB:
                count = counts[allele]
                if count > 0:
                    
                    weight = min(
                        total / (len(ALLELE_VOCAB) * count), 3.0
                    ) 
                else:
                    weight = 1.0
                weights.append(weight)

            weights = torch.tensor(weights, dtype=torch.float32)
            return weights

        counts1 = make_weights(counts1)
        counts2 = make_weights(counts2)

        return counts1, counts2

    def compute_accuracy(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        targets1: torch.Tensor,
        targets2: torch.Tensor,
    ) -> Dict[str, float]:
       
        correct1 = (pred1 == targets1).float()
        correct2 = (pred2 == targets2).float()

        allele1_acc = correct1.mean().item()
        allele2_acc = correct2.mean().item()
        both_acc = (correct1 * correct2).mean().item()

        return {
            "allele1_acc": allele1_acc,
            "allele2_acc": allele2_acc,
            "both_acc": both_acc,
        }

    def train_epoch(
        self,
        use_mixup: bool = False, 
        mixup_alpha: float = 0.2,
    ) -> Dict[str, float]:

        self.model.train()

        total_loss = 0.0
        all_pred1, all_pred2 = [], []
        all_targets1, all_targets2 = [], []

        for batch in self.train_loader:
            measurements, targets1, targets2 = batch
            measurements = measurements.to(self.device)
            targets1 = targets1.to(self.device)
            targets2 = targets2.to(self.device)

            self.optimizer.zero_grad()

            if use_mixup and np.random.random() > 0.75: 
                
                mixed_x, t1_a, t1_b, t2_a, t2_b, lam = self.mixup_data(
                    measurements, targets1, targets2, mixup_alpha
                )
                logits1, logits2 = self.model(mixed_x)

               
                loss1 = self.mixup_criterion(
                    self.criterion.criterion1, logits1, t1_a, t1_b, lam
                )
                loss2 = self.mixup_criterion(
                    self.criterion.criterion2, logits2, t2_a, t2_b, lam
                )
                loss = loss1 + loss2

                
                pred1 = logits1.argmax(dim=1)
                pred2 = logits2.argmax(dim=1)
                
                all_pred1.append(pred1.cpu())
                all_pred2.append(pred2.cpu())
                all_targets1.append(targets1.cpu())
                all_targets2.append(targets2.cpu())
            else:
                
                logits1, logits2 = self.model(measurements)
                loss, _, _ = self.criterion(logits1, logits2, targets1, targets2)

                all_pred1.append(logits1.argmax(dim=1).cpu())
                all_pred2.append(logits2.argmax(dim=1).cpu())
                all_targets1.append(targets1.cpu())
                all_targets2.append(targets2.cpu())

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()  
            total_loss += loss.item()

        n_batches = len(self.train_loader)
        all_pred1 = torch.cat(all_pred1)
        all_pred2 = torch.cat(all_pred2)
        all_targets1 = torch.cat(all_targets1)
        all_targets2 = torch.cat(all_targets2)

        acc = self.compute_accuracy(all_pred1, all_pred2, all_targets1, all_targets2)

        return {"loss": total_loss / n_batches, **acc}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
      
        self.model.eval()

        total_loss = 0.0
        all_pred1, all_pred2 = [], []
        all_targets1, all_targets2 = [], []

        for batch in self.val_loader:
            measurements, targets1, targets2 = batch
            measurements = measurements.to(self.device)
            targets1 = targets1.to(self.device)
            targets2 = targets2.to(self.device)

            logits1, logits2 = self.model(measurements)
            loss, _, _ = self.criterion(logits1, logits2, targets1, targets2)

            total_loss += loss.item()
            all_pred1.append(logits1.argmax(dim=1).cpu())
            all_pred2.append(logits2.argmax(dim=1).cpu())
            all_targets1.append(targets1.cpu())
            all_targets2.append(targets2.cpu())

        n_batches = len(self.val_loader)
        all_pred1 = torch.cat(all_pred1)
        all_pred2 = torch.cat(all_pred2)
        all_targets1 = torch.cat(all_targets1)
        all_targets2 = torch.cat(all_targets2)

        acc = self.compute_accuracy(all_pred1, all_pred2, all_targets1, all_targets2)

        return {"loss": total_loss / n_batches, **acc}

    def train_model(
        self, num_epochs: int, patience: int, save_path: Optional[str] = None
    ) -> Dict[str, list]:

        
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["both_acc"])
            history["val_acc"].append(val_metrics["both_acc"])

            cur_val = val_metrics["both_acc"]
            cur_train = train_metrics["both_acc"]

            
            improved = cur_val > best_val_acc or (
                cur_val == best_val_acc and cur_train > best_train_acc
            )

            marker = ""
            if improved:
                marker = " ✓"

            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {cur_train:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {cur_val:.4f} | "
                f"A1: {val_metrics['allele1_acc']:.4f}, A2: {val_metrics['allele2_acc']:.4f}{marker}"
            )

            if improved:
                best_val_acc = cur_val
                best_train_acc = cur_train
                best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0

                if save_path:
                    torch.save(
                        {
                            "model_state_dict": best_model_state,
                            "epoch": epoch,
                            "val_acc": best_val_acc,
                            "train_acc": best_train_acc,
                        },
                        save_path,
                    )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print("-" * 60)
        print(
            f"Best validation accuracy: {best_val_acc:.4f} (train: {best_train_acc:.4f})"
        )

        return history
