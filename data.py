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
from config import ALLELE_TO_IDX


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


class PCRDataPipeline:
    def __init__(
        self,
        filepath: str,
        batch_size: int,
        val_split: float,
        seed: int,
        augment_train: bool,
        seq_len: int = 46,
    ):
        self.filepath = filepath
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.augment_train = augment_train

        self.measurements = np.array([], dtype=np.float32)
        self.allele_pairs = None
        self.raw_genotypes = np.array([], dtype=np.float32)
        self.run_ids = np.array([], dtype=np.int32)
        self.is_control = None
        self.seq_len = seq_len

    def load_dataset(self):
        self.measurements = []
        self.allele_pairs = []
        self.raw_genotypes = []
        self.run_ids = []
        self.is_control = []

        with open(self.filepath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")

                if len(parts) < self.seq_len + 1:
                    continue

                meas = [float(x.replace(",", ".")) for x in parts[: self.seq_len]]
                self.measurements.append(meas)

                if len(parts) >= self.seq_len + 3:

                    genotype = parts[self.seq_len]
                    self.raw_genotypes.append(genotype)
                    self.allele_pairs.append(parse_genotype(genotype))
                    self.run_ids.append(int(parts[self.seq_len + 1]))
                    self.is_control.append(int(parts[self.seq_len + 2]))
                else:

                    self.raw_genotypes.append("")
                    self.allele_pairs.append(("", ""))
                    self.run_ids.append(0)
                    self.is_control.append(int(parts[self.seq_len + 2]))

        self.measurements = np.array(self.measurements, dtype=np.float32)
        self.run_ids = np.array(self.run_ids, dtype=np.float32)
        self.is_control = np.array(self.is_control, dtype=np.int32)

        return (
            self.measurements,
            self.allele_pairs,
            self.raw_genotypes,
            self.run_ids,
            self.is_control,
        )

    def build_control_lookup(self):
        is_control = self.is_control

        control_lookup = {}
        control_indices = np.where(is_control == 1)[0]
        run_ids = self.run_ids
        measurements = self.measurements

        for idx in control_indices:
            run_id = run_ids[idx]
            control_lookup[run_id] = measurements[idx].copy()

        return control_lookup

    def build_control_pools(self):

        measurements = self.measurements
        raw_genotypes = self.raw_genotypes
        run_ids = self.run_ids
        is_control = self.is_control

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

    def create_data_loaders(self):
        filepath = self.filepath
        batch_size = self.batch_size
        val_split = self.val_split
        seed = self.seed
        augment_train = self.augment_train

        measurements, allele_pairs, raw_genotypes, run_ids, is_control = (
            self.load_dataset()
        )

        control_78_lookup = self.build_control_lookup()

        control_77_lookup = {}
        for i in range(len(measurements)):
            run_id = int(run_ids[i])
            if raw_genotypes[i] == "7_7" and run_id not in control_77_lookup:
                control_77_lookup[run_id] = measurements[i].copy()

        pool_78, pool_77 = self.build_control_pools()

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

        print(
            f"Dataset loaded: {len(measurements)} total samples, {len(unique_runs)} runs"
        )
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


def load_inference_data(filepath: str, seq_len: int = 46):

    pipeline = PCRDataPipeline(
        filepath,
        batch_size=1,
        val_split=0,
        seed=42,
        augment_train=False,
        seq_len=seq_len,
    )
    measurements, _, _, _, is_control = pipeline.load_dataset()

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
