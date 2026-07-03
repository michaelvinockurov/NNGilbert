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
if torch.cuda.is_available() == False:
    exit


def set_seed(seed: int = RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_derivatives(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    first_deriv = np.gradient(signal, axis=1)

    second_deriv = np.gradient(first_deriv, axis=1)

    return first_deriv, second_deriv


def parse_genotype(genotype_str: str) -> Tuple[str, str]:

    if genotype_str == "K-":
        return ("K-", "K-")

    parts = genotype_str.split("_")
    allele1, allele2 = parts[0], parts[1]

    if int(allele1) > int(allele2):
        allele1, allele2 = allele2, allele1

    return (allele1, allele2)


def load_dataset(
    filepath: str,
) -> Tuple[np.ndarray, List[Tuple[str, str]], List[str], np.ndarray, np.ndarray]:

    measurements = []
    allele_pairs = []
    raw_genotypes = []
    run_ids = []
    is_control = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) < 47:
                continue

            meas = [float(x) for x in parts[:46]]
            measurements.append(meas)

            if len(parts) >= 49:

                genotype = parts[46]
                raw_genotypes.append(genotype)
                allele_pairs.append(parse_genotype(genotype))
                run_ids.append(int(parts[47]))
                is_control.append(int(parts[48]))
            else:

                raw_genotypes.append("")
                allele_pairs.append(("", ""))
                run_ids.append(0)
                is_control.append(int(parts[46]))

    return (
        np.array(measurements, dtype=np.float32),
        allele_pairs,
        raw_genotypes,
        np.array(run_ids, dtype=np.int32),
        np.array(is_control, dtype=np.int32),
    )


def build_control_lookup(
    measurements: np.ndarray, run_ids: np.ndarray, is_control: np.ndarray
) -> Dict[int, np.ndarray]:

    control_lookup = {}
    control_indices = np.where(is_control == 1)[0]

    for idx in control_indices:
        run_id = run_ids[idx]
        control_lookup[run_id] = measurements[idx].copy()

    return control_lookup


def build_control_pools(
    measurements: np.ndarray,
    raw_genotypes: List[str],
    run_ids: np.ndarray,
    is_control: np.ndarray,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    pool_78 = {}
    pool_77 = {}

    for i in range(len(measurements)):
        run_id = int(run_ids[i])
        genotype = raw_genotypes[i]

        if genotype == "7_8":
            if run_id not in pool_78:
                pool_78[run_id] = []
            pool_78[run_id].append(measurements[i].copy())
        elif genotype == "7_7":
            if run_id not in pool_77:
                pool_77[run_id] = []
            pool_77[run_id].append(measurements[i].copy())

    return pool_78, pool_77


class MeltingCurveDataset(Dataset):

    def __init__(
        self,
        measurements: np.ndarray,
        allele_pairs: List[Tuple[str, str]],
        run_ids: np.ndarray,
        control_78_lookup: Dict[int, np.ndarray],
        control_77_lookup: Dict[int, np.ndarray],
        pool_78: Optional[Dict[int, List[np.ndarray]]] = None,
        pool_77: Optional[Dict[int, List[np.ndarray]]] = None,
        normalize: bool = True,
        augment: bool = False,
        noise_std: float = 0.02,
        control_noise_std: float = 0.01,
    ):

        self.raw_measurements = measurements.copy()
        self.allele_pairs = allele_pairs
        self.run_ids = run_ids
        self.control_78_lookup = control_78_lookup
        self.control_77_lookup = control_77_lookup
        self.pool_78 = pool_78 or {}
        self.pool_77 = pool_77 or {}
        self.normalize = normalize
        self.augment = augment
        self.noise_std = noise_std
        self.control_noise_std = control_noise_std

        self.allele1_indices = torch.tensor(
            [ALLELE_TO_IDX[a1] for a1, _ in allele_pairs], dtype=torch.long
        )
        self.allele2_indices = torch.tensor(
            [ALLELE_TO_IDX[a2] for _, a2 in allele_pairs], dtype=torch.long
        )

        self.data = self._prepare_all_data()

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:

        mean = signal.mean()
        std = signal.std()
        if std < 1e-8:
            std = 1.0
        return (signal - mean) / std

    def _get_controls(
        self, run_id: int, rotate: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        if rotate and run_id in self.pool_78 and len(self.pool_78[run_id]) > 0:
            ctrl_78 = self.pool_78[run_id][np.random.randint(len(self.pool_78[run_id]))]
        else:
            ctrl_78 = self.control_78_lookup.get(run_id)

        if rotate and run_id in self.pool_77 and len(self.pool_77[run_id]) > 0:
            ctrl_77 = self.pool_77[run_id][np.random.randint(len(self.pool_77[run_id]))]
        else:
            ctrl_77 = self.control_77_lookup.get(run_id)

        return ctrl_78, ctrl_77

    def _prepare_sample(
        self,
        sample_meas: np.ndarray,
        ctrl_78_meas: np.ndarray,
        ctrl_77_meas: np.ndarray,
    ) -> torch.Tensor:

        sample_d1, sample_d2 = compute_derivatives(sample_meas.reshape(1, -1))
        sample_d1, sample_d2 = sample_d1.squeeze(), sample_d2.squeeze()

        ctrl_78_d1, _ = compute_derivatives(ctrl_78_meas.reshape(1, -1))
        ctrl_78_d1 = ctrl_78_d1.squeeze()

        ctrl_77_d1, _ = compute_derivatives(ctrl_77_meas.reshape(1, -1))
        ctrl_77_d1 = ctrl_77_d1.squeeze()

        diff_78 = sample_meas - ctrl_78_meas
        diff_78_d1 = sample_d1 - ctrl_78_d1
        diff_77 = sample_meas - ctrl_77_meas
        diff_77_d1 = sample_d1 - ctrl_77_d1

        if self.normalize:
            norm = self._normalize_signal
            channels = np.stack(
                [
                    norm(sample_meas),
                    norm(sample_d1),
                    norm(sample_d2),
                    norm(diff_78),
                    norm(diff_78_d1),
                    norm(ctrl_78_meas),
                    norm(ctrl_78_d1),
                    norm(diff_77),
                    norm(diff_77_d1),
                    norm(ctrl_77_meas),
                    norm(ctrl_77_d1),
                ],
                axis=0,
            )
        else:
            channels = np.stack(
                [
                    sample_meas,
                    sample_d1,
                    sample_d2,
                    diff_78,
                    diff_78_d1,
                    ctrl_78_meas,
                    ctrl_78_d1,
                    diff_77,
                    diff_77_d1,
                    ctrl_77_meas,
                    ctrl_77_d1,
                ],
                axis=0,
            )

        return torch.tensor(channels, dtype=torch.float32)

    def _prepare_all_data(self) -> List[torch.Tensor]:

        data = []
        for i in range(len(self.raw_measurements)):
            run_id = self.run_ids[i]
            ctrl_78, ctrl_77 = self._get_controls(run_id, rotate=False)

            if ctrl_78 is None:
                ctrl_78 = self.raw_measurements[i]
            if ctrl_77 is None:
                ctrl_77 = self.raw_measurements[i]
            tensor = self._prepare_sample(self.raw_measurements[i], ctrl_78, ctrl_77)
            data.append(tensor)
        return data

    def _augment_signal(self, signal: np.ndarray, noise_std: float) -> np.ndarray:

        aug = signal.copy()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std * aug.std(), aug.shape)
            aug = aug + noise
        return aug

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        run_id = self.run_ids[idx]

        if self.augment or len(self.pool_78) > 0:
            ctrl_78, ctrl_77 = self._get_controls(run_id, rotate=True)
            if ctrl_78 is None:
                ctrl_78 = self.raw_measurements[idx]
            if ctrl_77 is None:
                ctrl_77 = self.raw_measurements[idx]

            sample = self.raw_measurements[idx]
            if self.augment:
                sample = self._augment_signal(sample, self.noise_std)
                ctrl_78 = self._augment_signal(ctrl_78, self.control_noise_std)
                ctrl_77 = self._augment_signal(ctrl_77, self.control_noise_std)

            data = self._prepare_sample(sample, ctrl_78, ctrl_77)
        else:
            data = self.data[idx]

        return (data, self.allele1_indices[idx], self.allele2_indices[idx])


def create_data_loaders(
    filepath: str,
    batch_size: int = BATCH_SIZE,
    val_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
    augment_train: bool = False,
) -> Tuple[
    DataLoader,
    DataLoader,
    List[str],
    List[str],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
]:

    measurements, allele_pairs, raw_genotypes, run_ids, is_control = load_dataset(
        filepath
    )

    control_78_lookup = build_control_lookup(measurements, run_ids, is_control)

    control_77_lookup = {}
    for i in range(len(measurements)):
        run_id = int(run_ids[i])
        if raw_genotypes[i] == "7_7" and run_id not in control_77_lookup:
            control_77_lookup[run_id] = measurements[i].copy()

    pool_78, pool_77 = build_control_pools(
        measurements, raw_genotypes, run_ids, is_control
    )

    unique_runs = np.unique(run_ids)

    train_indices = []
    val_indices = []

    np.random.seed(seed)

    for run_id in unique_runs:

        run_mask = (run_ids == run_id) & (is_control == 0)
        run_indices = np.where(run_mask)[0]

        if len(run_indices) == 0:
            continue

        run_genotypes = [raw_genotypes[i] for i in run_indices]

        try:
            train_idx, val_idx = train_test_split(
                run_indices,
                test_size=val_split,
                random_state=seed,
                stratify=run_genotypes,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                run_indices, test_size=val_split, random_state=seed
            )

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    train_idx = np.array(train_indices)
    val_idx = np.array(val_indices)

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_genotypes = [raw_genotypes[i] for i in train_idx]
    val_genotypes = [raw_genotypes[i] for i in val_idx]

    train_dataset = MeltingCurveDataset(
        measurements[train_idx],
        [allele_pairs[i] for i in train_idx],
        run_ids[train_idx],
        control_78_lookup,
        control_77_lookup,
        pool_78=pool_78,
        pool_77=pool_77,
        normalize=True,
        augment=augment_train,
    )

    val_dataset = MeltingCurveDataset(
        measurements[val_idx],
        [allele_pairs[i] for i in val_idx],
        run_ids[val_idx],
        control_78_lookup,
        control_77_lookup,
        pool_78=None,
        pool_77=None,
        normalize=True,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset loaded: {len(measurements)} total samples, {len(unique_runs)} runs")
    print(f"  7_8 controls: {len(control_78_lookup)} runs")
    print(f"  7_7 controls: {len(control_77_lookup)} runs")
    print(
        f"  Control rotation pools: 7_8={sum(len(v) for v in pool_78.values())} samples, "
        f"7_7={sum(len(v) for v in pool_77.values())} samples"
    )
    print(f"  Training:   {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples")

    return (
        train_loader,
        val_loader,
        train_genotypes,
        val_genotypes,
        control_78_lookup,
        control_77_lookup,
    )


class ConvBlock(nn.Module):
    """Convolutional block with Conv1d -> BatchNorm -> ReLU -> Dropout."""

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
        """Get predicted class indices."""
        logits1, logits2 = self.forward(x)
        pred1 = logits1.argmax(dim=1)
        pred2 = logits2.argmax(dim=1)
        return pred1, pred2


def mixup_data(
    x: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:

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
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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


def compute_class_weights(genotypes: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute separate class weights for each allele head."""
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

                weight = min(total / (len(ALLELE_VOCAB) * count), 3.0)
            else:
                weight = 1.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)

    return make_weights(counts1), make_weights(counts2)


def compute_accuracy(
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
    model: DualHeadCNN,
    train_loader: DataLoader,
    criterion: DualHeadLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    all_pred1, all_pred2 = [], []
    all_targets1, all_targets2 = [], []

    for batch in train_loader:
        measurements, targets1, targets2 = batch
        measurements = measurements.to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)

        optimizer.zero_grad()

        if use_mixup and np.random.random() > 0.75:

            mixed_x, t1_a, t1_b, t2_a, t2_b, lam = mixup_data(
                measurements, targets1, targets2, mixup_alpha
            )
            logits1, logits2 = model(mixed_x)

            loss1 = mixup_criterion(criterion.criterion1, logits1, t1_a, t1_b, lam)
            loss2 = mixup_criterion(criterion.criterion2, logits2, t2_a, t2_b, lam)
            loss = loss1 + loss2

            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)

            all_pred1.append(pred1.cpu())
            all_pred2.append(pred2.cpu())
            all_targets1.append(targets1.cpu())
            all_targets2.append(targets2.cpu())
        else:

            logits1, logits2 = model(measurements)
            loss, _, _ = criterion(logits1, logits2, targets1, targets2)

            all_pred1.append(logits1.argmax(dim=1).cpu())
            all_pred2.append(logits2.argmax(dim=1).cpu())
            all_targets1.append(targets1.cpu())
            all_targets2.append(targets2.cpu())

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    n_batches = len(train_loader)
    all_pred1 = torch.cat(all_pred1)
    all_pred2 = torch.cat(all_pred2)
    all_targets1 = torch.cat(all_targets1)
    all_targets2 = torch.cat(all_targets2)

    acc = compute_accuracy(all_pred1, all_pred2, all_targets1, all_targets2)

    return {"loss": total_loss / n_batches, **acc}


@torch.no_grad()
def validate(
    model: DualHeadCNN,
    val_loader: DataLoader,
    criterion: DualHeadLoss,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    total_loss = 0.0
    all_pred1, all_pred2 = [], []
    all_targets1, all_targets2 = [], []

    for batch in val_loader:
        measurements, targets1, targets2 = batch
        measurements = measurements.to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)

        logits1, logits2 = model(measurements)
        loss, _, _ = criterion(logits1, logits2, targets1, targets2)

        total_loss += loss.item()
        all_pred1.append(logits1.argmax(dim=1).cpu())
        all_pred2.append(logits2.argmax(dim=1).cpu())
        all_targets1.append(targets1.cpu())
        all_targets2.append(targets2.cpu())

    n_batches = len(val_loader)
    all_pred1 = torch.cat(all_pred1)
    all_pred2 = torch.cat(all_pred2)
    all_targets1 = torch.cat(all_targets1)
    all_targets2 = torch.cat(all_targets2)

    acc = compute_accuracy(all_pred1, all_pred2, all_targets1, all_targets2)

    return {"loss": total_loss / n_batches, **acc}


def train_model(
    dataset_path: str,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: torch.device = DEVICE,
    save_path: Optional[str] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[
    DualHeadCNN, Dict[str, List[float]], Dict[int, np.ndarray], Dict[int, np.ndarray]
]:

    set_seed(seed)

    print(f"Training on: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("-" * 60)

    (
        train_loader,
        val_loader,
        train_genotypes,
        _,
        control_78_lookup,
        control_77_lookup,
    ) = create_data_loaders(dataset_path, batch_size=batch_size, augment_train=False)

    weights1, weights2 = compute_class_weights(train_genotypes)
    weights1, weights2 = weights1.to(device), weights2.to(device)
    print(f"Allele 1 weights: {dict(zip(ALLELE_VOCAB, weights1.tolist()))}")
    print(f"Allele 2 weights: {dict(zip(ALLELE_VOCAB, weights2.tolist()))}")
    print("-" * 60)

    model = DualHeadCNN(input_channels=11).to(device)

    criterion = DualHeadLoss(weights1=weights1, weights2=weights2, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

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
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_metrics = validate(model, val_loader, criterion, device)

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
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0

            if save_path:
                torch.save(
                    {
                        "model_state_dict": best_model_state,
                        "epoch": epoch,
                        "val_acc": best_val_acc,
                        "train_acc": best_train_acc,
                        "weights1": weights1.cpu(),
                        "weights2": weights2.cpu(),
                        "control_78_lookup": {
                            k: v for k, v in control_78_lookup.items()
                        },
                        "control_77_lookup": {
                            k: v for k, v in control_77_lookup.items()
                        },
                    },
                    save_path,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f} (train: {best_train_acc:.4f})")

    return model, history, control_78_lookup, control_77_lookup


def train_ensemble(
    dataset_path: str, n_models: int = 3, base_seed: int = RANDOM_SEED, **train_kwargs
) -> Tuple[List[DualHeadCNN], Dict[int, np.ndarray], Dict[int, np.ndarray]]:

    models = []
    control_78_lookup: Dict[int, np.ndarray] = {}
    control_77_lookup: Dict[int, np.ndarray] = {}

    for i in range(n_models):
        seed = base_seed + i * 137
        print(f"\n{'='*60}")
        print(f"Training ensemble model {i+1}/{n_models} (seed={seed})")
        print(f"{'='*60}")

        set_seed(seed)

        model, _, control_78_lookup, control_77_lookup = train_model(
            dataset_path=dataset_path, seed=seed, **train_kwargs
        )
        models.append(model)

    return models, control_78_lookup, control_77_lookup


def predict_with_ensemble(
    models: List[DualHeadCNN],
    sample_measurements: np.ndarray,
    control_78_measurements: np.ndarray,
    control_77_measurements: np.ndarray,
    device: torch.device = DEVICE,
    use_tta: bool = True,
    tta_runs: int = 3,
) -> Tuple[str, Tuple[float, float]]:

    all_probs1 = []
    all_probs2 = []

    for model in models:
        model.eval()
        with torch.no_grad():
            if use_tta:

                model_probs1, model_probs2 = [], []

                x = prepare_sample_for_inference(
                    sample_measurements,
                    control_78_measurements,
                    control_77_measurements,
                    add_noise=False,
                )
                x = x.to(device)
                logits1, logits2 = model(x)
                model_probs1.append(torch.softmax(logits1, dim=1))
                model_probs2.append(torch.softmax(logits2, dim=1))

                for _ in range(tta_runs - 1):
                    x = prepare_sample_for_inference(
                        sample_measurements,
                        control_78_measurements,
                        control_77_measurements,
                        add_noise=True,
                        noise_std=0.01,
                    )
                    x = x.to(device)
                    logits1, logits2 = model(x)
                    model_probs1.append(torch.softmax(logits1, dim=1))
                    model_probs2.append(torch.softmax(logits2, dim=1))

                avg_probs1 = torch.stack(model_probs1).mean(dim=0)
                avg_probs2 = torch.stack(model_probs2).mean(dim=0)
            else:
                x = prepare_sample_for_inference(
                    sample_measurements,
                    control_78_measurements,
                    control_77_measurements,
                )
                x = x.to(device)
                logits1, logits2 = model(x)
                avg_probs1 = torch.softmax(logits1, dim=1)
                avg_probs2 = torch.softmax(logits2, dim=1)

            all_probs1.append(avg_probs1)
            all_probs2.append(avg_probs2)

    probs1 = torch.stack(all_probs1).mean(dim=0)
    probs2 = torch.stack(all_probs2).mean(dim=0)

    pred1 = probs1.argmax(dim=1).item()
    pred2 = probs2.argmax(dim=1).item()

    conf1 = probs1[0, pred1].item()
    conf2 = probs2[0, pred2].item()

    allele1 = IDX_TO_ALLELE[pred1]
    allele2 = IDX_TO_ALLELE[pred2]

    if allele1 == "K-" or allele2 == "K-":
        genotype = "K-"
    else:
        a1, a2 = int(allele1), int(allele2)
        if a1 > a2:
            a1, a2 = a2, a1
            conf1, conf2 = conf2, conf1
        genotype = f"{a1}_{a2}"

    return genotype, (conf1, conf2)


def prepare_sample_for_inference(
    sample_meas: np.ndarray,
    control_78_meas: np.ndarray,
    control_77_meas: np.ndarray,
    add_noise: bool = False,
    noise_std: float = 0.01,
) -> torch.Tensor:

    sample_meas = sample_meas.flatten().copy()
    control_78_meas = control_78_meas.flatten().copy()
    control_77_meas = control_77_meas.flatten().copy()

    if add_noise:
        sample_meas += np.random.normal(
            0, noise_std * sample_meas.std(), sample_meas.shape
        )
        control_78_meas += np.random.normal(
            0, noise_std * control_78_meas.std(), control_78_meas.shape
        )
        control_77_meas += np.random.normal(
            0, noise_std * control_77_meas.std(), control_77_meas.shape
        )

    sample_d1, sample_d2 = compute_derivatives(sample_meas.reshape(1, -1))
    sample_d1, sample_d2 = sample_d1.squeeze(), sample_d2.squeeze()

    ctrl_78_d1, _ = compute_derivatives(control_78_meas.reshape(1, -1))
    ctrl_78_d1 = ctrl_78_d1.squeeze()

    ctrl_77_d1, _ = compute_derivatives(control_77_meas.reshape(1, -1))
    ctrl_77_d1 = ctrl_77_d1.squeeze()

    diff_78 = sample_meas - control_78_meas
    diff_78_d1 = sample_d1 - ctrl_78_d1
    diff_77 = sample_meas - control_77_meas
    diff_77_d1 = sample_d1 - ctrl_77_d1

    def normalize(arr):
        mean, std = arr.mean(), arr.std()
        return (arr - mean) / (std if std > 1e-8 else 1.0)

    channels = np.stack(
        [
            normalize(sample_meas),
            normalize(sample_d1),
            normalize(sample_d2),
            normalize(diff_78),
            normalize(diff_78_d1),
            normalize(control_78_meas),
            normalize(ctrl_78_d1),
            normalize(diff_77),
            normalize(diff_77_d1),
            normalize(control_77_meas),
            normalize(ctrl_77_d1),
        ],
        axis=0,
    )

    return torch.tensor(channels, dtype=torch.float32).unsqueeze(0)


def predict_genotype(
    model: DualHeadCNN,
    sample_measurements: np.ndarray,
    control_78_measurements: np.ndarray,
    control_77_measurements: np.ndarray,
    device: torch.device = DEVICE,
    use_tta: bool = True,
    tta_runs: int = 5,
) -> Tuple[str, Tuple[float, float]]:

    model.eval()

    with torch.no_grad():
        if use_tta:

            all_probs1, all_probs2 = [], []

            x = prepare_sample_for_inference(
                sample_measurements,
                control_78_measurements,
                control_77_measurements,
                add_noise=False,
            )
            x = x.to(device)
            logits1, logits2 = model(x)
            all_probs1.append(torch.softmax(logits1, dim=1))
            all_probs2.append(torch.softmax(logits2, dim=1))

            for _ in range(tta_runs - 1):
                x = prepare_sample_for_inference(
                    sample_measurements,
                    control_78_measurements,
                    control_77_measurements,
                    add_noise=True,
                    noise_std=0.01,
                )
                x = x.to(device)
                logits1, logits2 = model(x)
                all_probs1.append(torch.softmax(logits1, dim=1))
                all_probs2.append(torch.softmax(logits2, dim=1))

            probs1 = torch.stack(all_probs1).mean(dim=0)
            probs2 = torch.stack(all_probs2).mean(dim=0)
        else:
            x = prepare_sample_for_inference(
                sample_measurements, control_78_measurements, control_77_measurements
            )
            x = x.to(device)
            logits1, logits2 = model(x)
            probs1 = torch.softmax(logits1, dim=1)
            probs2 = torch.softmax(logits2, dim=1)

        pred1 = probs1.argmax(dim=1).item()
        pred2 = probs2.argmax(dim=1).item()

        conf1 = probs1[0, pred1].item()
        conf2 = probs2[0, pred2].item()

    allele1 = IDX_TO_ALLELE[pred1]
    allele2 = IDX_TO_ALLELE[pred2]

    if allele1 == "K-" or allele2 == "K-":
        genotype = "K-"
    else:
        a1, a2 = int(allele1), int(allele2)
        if a1 > a2:
            a1, a2 = a2, a1
            conf1, conf2 = conf2, conf1
        genotype = f"{a1}_{a2}"

    return genotype, (conf1, conf2)


def predict_batch_samples(
    model: DualHeadCNN,
    sample_measurements: np.ndarray,
    control_78_measurements: np.ndarray,
    control_77_measurements: np.ndarray,
    device: torch.device = DEVICE,
) -> List[Tuple[str, Tuple[float, float]]]:

    model.eval()

    channel_tensors = []
    for i in range(len(sample_measurements)):
        channels = prepare_sample_for_inference(
            sample_measurements[i], control_78_measurements, control_77_measurements
        )
        channel_tensors.append(channels)

    x = torch.cat(channel_tensors, dim=0).to(device)

    results = []
    with torch.no_grad():
        logits1, logits2 = model(x)
        probs1 = torch.softmax(logits1, dim=1)
        probs2 = torch.softmax(logits2, dim=1)

        preds1 = logits1.argmax(dim=1)
        preds2 = logits2.argmax(dim=1)

        for i in range(len(sample_measurements)):
            pred1 = preds1[i].item()
            pred2 = preds2[i].item()
            conf1 = probs1[i, pred1].item()
            conf2 = probs2[i, pred2].item()

            allele1 = IDX_TO_ALLELE[pred1]
            allele2 = IDX_TO_ALLELE[pred2]

            if allele1 == "K-" or allele2 == "K-":
                genotype = "K-"
            else:
                a1, a2 = int(allele1), int(allele2)
                if a1 > a2:
                    a1, a2 = a2, a1
                    conf1, conf2 = conf2, conf1
                genotype = f"{a1}_{a2}"

            results.append((genotype, (conf1, conf2)))

    return results


def load_model(
    checkpoint_path: str, device: torch.device = DEVICE
) -> Tuple[DualHeadCNN, Dict[int, np.ndarray]]:

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DualHeadCNN(input_channels=11).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    control_lookup = checkpoint.get("control_lookup", {})

    print(f"Model loaded from {checkpoint_path}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")

    return model, control_lookup


def save_ensemble(
    models: List[DualHeadCNN],
    control_78_lookup: Dict[int, np.ndarray],
    control_77_lookup: Dict[int, np.ndarray],
    output_dir: str,
    val_acc: float = 0.0,
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(models):
        model_path = output_path / f"model_{i}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            },
            model_path,
        )

    meta_path = output_path / "ensemble_meta.pt"
    torch.save(
        {
            "n_models": len(models),
            "control_78_lookup": {k: v for k, v in control_78_lookup.items()},
            "control_77_lookup": {k: v for k, v in control_77_lookup.items()},
            "val_acc": val_acc,
        },
        meta_path,
    )

    print(f"Ensemble saved to {output_dir} ({len(models)} models)")


def load_ensemble(
    model_dir: str, device: torch.device = DEVICE
) -> Tuple[List[DualHeadCNN], Dict[int, np.ndarray], Dict[int, np.ndarray]]:

    model_path = Path(model_dir)

    meta_path = model_path / "ensemble_meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Ensemble metadata not found: {meta_path}")

    meta = torch.load(meta_path, map_location=device, weights_only=False)
    n_models = meta["n_models"]
    control_78_lookup = meta.get("control_78_lookup", meta.get("control_lookup", {}))
    control_77_lookup = meta.get("control_77_lookup", {})

    models = []
    for i in range(n_models):
        model_file = model_path / f"model_{i}.pt"
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)

        model = DualHeadCNN(input_channels=11).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    print(f"Loaded ensemble from {model_dir} ({n_models} models)")
    print(f"  Validation accuracy: {meta.get('val_acc', 'N/A'):.4f}")

    return models, control_78_lookup, control_77_lookup


def load_inference_data(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    measurements, _, _, _, is_control = load_dataset(filepath)

    ctrl_78_mask = is_control == 1
    ctrl_77_mask = is_control == 2
    sample_mask = is_control == 0

    if not np.any(ctrl_78_mask):
        raise ValueError("No 7_8 control sample found in inference data (is_control=1)")
    if not np.any(ctrl_77_mask):
        raise ValueError("No 7_7 control sample found in inference data (is_control=2)")

    control_78_measurements = measurements[ctrl_78_mask][0]
    control_77_measurements = measurements[ctrl_77_mask][0]
    sample_measurements = measurements[sample_mask]
    sample_indices = np.where(sample_mask)[0]

    return (
        sample_measurements,
        control_78_measurements,
        control_77_measurements,
        sample_indices,
    )


def run_inference(
    models: List[DualHeadCNN],
    data_path: str,
    output_path: Optional[str] = None,
    device: torch.device = DEVICE,
) -> List[Dict]:

    print(f"\nLoading inference data from: {data_path}")
    sample_meas, control_78_meas, control_77_meas, sample_indices = load_inference_data(
        data_path
    )

    print(f"  Samples to classify: {len(sample_meas)}")
    print(f"  7_8 control found: 1")
    print(f"  7_7 control found: 1")
    print()

    results = []

    for i, idx in enumerate(sample_indices):
        pred_genotype, confidences = predict_with_ensemble(
            models,
            sample_meas[i],
            control_78_meas,
            control_77_meas,
            device=device,
            use_tta=True,
            tta_runs=5,
        )

        results.append(
            {
                "row": idx + 1,
                "genotype": pred_genotype,
                "conf_allele1": confidences[0],
                "conf_allele2": confidences[1],
                "min_conf": min(confidences),
            }
        )

    print("Predictions:")
    print("-" * 60)
    for r in results:
        conf_flag = (
            "" if r["min_conf"] > 0.8 else " ⚠️" if r["min_conf"] > 0.65 else " ❌"
        )
        print(
            f"  Row {r['row']:3d}: {r['genotype']:5s}  (conf: {r['conf_allele1']:.2f}, {r['conf_allele2']:.2f}){conf_flag}"
        )

    print("-" * 60)
    genotype_counts = {}
    for r in results:
        g = r["genotype"]
        genotype_counts[g] = genotype_counts.get(g, 0) + 1

    print("Summary:")
    for g in sorted(genotype_counts.keys()):
        print(f"  {g}: {genotype_counts[g]} samples")

    low_conf = sum(1 for r in results if r["min_conf"] < 0.9)
    if low_conf > 0:
        print(f"\n⚠️ {low_conf} samples with confidence < 0.9")

    if output_path:
        with open(output_path, "w") as f:
            f.write("row\tgenotype\tconf_allele1\tconf_allele2\n")
            for r in results:
                f.write(
                    f"{r['row']}\t{r['genotype']}\t{r['conf_allele1']:.4f}\t{r['conf_allele2']:.4f}\n"
                )
        print(f"\nResults saved to: {output_path}")

    return results


def parse_args():

    parser = argparse.ArgumentParser(
        description="PCR Melting Curve Genotype Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train:    python model.py train --data dataset.tsv --output models/
  Infer:    python model.py infer --data testing_data.tsv --model models/ --output results.tsv
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    train_parser = subparsers.add_parser("train", help="Train model(s)")
    train_parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="dataset.tsv",
        help="Path to training dataset TSV (default: dataset.tsv)",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models",
        help="Output directory for trained models (default: models/)",
    )
    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    train_parser.add_argument(
        "--ensemble",
        "-n",
        type=int,
        default=5,
        help="Number of models in ensemble (default: 5, use 1 for single model)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    train_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference on new samples")
    infer_parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to inference data TSV (required)",
    )
    infer_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models",
        help="Path to model directory (default: models/)",
    )
    infer_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output results TSV (optional)",
    )

    return parser.parse_args()


def cmd_train(args):

    dataset_path = Path(args.data)
    output_dir = Path(args.output)

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return 1

    print("=" * 60)
    print("PCR Melting Curve Genotype Classification - Training")
    print("=" * 60)
    print(f"Dataset:  {dataset_path}")
    print(f"Output:   {output_dir}")
    print(f"Ensemble: {args.ensemble} models")
    print(f"Epochs:   {args.epochs}")
    print(f"Device:   {DEVICE}")
    print("=" * 60)

    if args.ensemble > 1:

        models, control_78_lookup, control_77_lookup = train_ensemble(
            dataset_path=str(dataset_path),
            n_models=args.ensemble,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

        save_ensemble(models, control_78_lookup, control_77_lookup, str(output_dir))
    else:

        model, history, control_78_lookup, control_77_lookup = train_model(
            dataset_path=str(dataset_path),
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
        models = [model]
        save_ensemble(models, control_78_lookup, control_77_lookup, str(output_dir))

    print("\n" + "=" * 60)
    print("Validation on Training Dataset")
    print("=" * 60)

    measurements, allele_pairs, raw_genotypes, run_ids, is_control = load_dataset(
        str(dataset_path)
    )

    unique_runs = np.unique(run_ids)
    np.random.seed(RANDOM_SEED)

    val_indices_set = set()
    for run_id in unique_runs:
        run_mask = (run_ids == run_id) & (is_control == 0)
        run_indices = np.where(run_mask)[0]
        if len(run_indices) == 0:
            continue
        run_genotypes = [raw_genotypes[i] for i in run_indices]
        try:
            _, val_idx = train_test_split(
                run_indices,
                test_size=VALIDATION_SPLIT,
                random_state=RANDOM_SEED,
                stratify=run_genotypes,
            )
        except ValueError:
            _, val_idx = train_test_split(
                run_indices, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
            )
        val_indices_set.update(val_idx)

    control_78_lookup_val = build_control_lookup(measurements, run_ids, is_control)
    control_77_lookup_val = {}
    for i in range(len(measurements)):
        rid = int(run_ids[i])
        if raw_genotypes[i] == "7_7" and rid not in control_77_lookup_val:
            control_77_lookup_val[rid] = measurements[i].copy()

    train_correct, train_total = 0, 0
    val_correct, val_total = 0, 0
    misclassified = []
    y_true_val = []
    y_pred_val = []

    for i in range(len(measurements)):
        if is_control[i]:
            continue

        run_id = run_ids[i]
        control_78_meas = control_78_lookup_val.get(run_id)
        control_77_meas = control_77_lookup_val.get(run_id)
        if control_78_meas is None or control_77_meas is None:
            continue

        pred_genotype, confidences = predict_with_ensemble(
            models,
            measurements[i],
            control_78_meas,
            control_77_meas,
            use_tta=True,
            tta_runs=5,
        )

        true_genotype = raw_genotypes[i]
        is_val = i in val_indices_set

        if is_val:
            val_total += 1
            y_true_val.append(true_genotype)
            y_pred_val.append(pred_genotype)
            if pred_genotype == true_genotype:
                val_correct += 1

        else:
            train_total += 1
            if pred_genotype == true_genotype:
                train_correct += 1

        if pred_genotype != true_genotype:
            misclassified.append(
                {
                    "index": i,
                    "run_id": run_id,
                    "is_val": is_val,
                    "true": true_genotype,
                    "pred": pred_genotype,
                    "conf": confidences,
                }
            )

    return 0


def cmd_infer(args):

    model_dir = Path(args.model)
    data_path = Path(args.data)

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return 1

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    print("=" * 60)
    print("PCR Melting Curve Genotype Classification - Inference")
    print("=" * 60)
    print(f"Model:  {model_dir}")
    print(f"Data:   {data_path}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    models, _, _ = load_ensemble(str(model_dir), device=DEVICE)

    results = run_inference(models, str(data_path), args.output, device=DEVICE)

    return 0


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        exit(cmd_train(args))
    elif args.mode == "infer":
        exit(cmd_infer(args))
    else:

        print("Usage: python model.py {train|infer} [options]")
        print("       python model.py train -h    # Training options")
        print("       python model.py infer -h    # Inference options")
        exit(1)
