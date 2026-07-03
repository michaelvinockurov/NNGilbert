import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

ALLELE_VOCAB = ["5", "6", "7", "8", "K-"]
IDX_TO_ALLELE = {i: a for i, a in enumerate(ALLELE_VOCAB)}
NUM_ALLELE_CLASSES = len(ALLELE_VOCAB)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MEASUREMENTS = 45
TTA_RUNS = 5


def compute_derivatives(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    d1 = np.gradient(signal, axis=1)
    d2 = np.gradient(d1, axis=1)
    return d1, d2


def normalize(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    return (arr - arr.mean()) / (std if std > 1e-8 else 1.0)


class ConvBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.3
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualHeadCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 11,
        num_classes: int = NUM_ALLELE_CLASSES,
        dropout: float = 0.15,
    ):
        super().__init__()
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
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x).squeeze(-1)
        s = self.shared_fc(f)
        return self.head1(s), self.head2(s)


def load_ensemble(model_dir: str) -> List[DualHeadCNN]:
    model_path = Path(model_dir)
    meta_path = model_path / "ensemble_meta.pt"
    if not meta_path.exists():
        print(f"Error: ensemble metadata not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    meta = torch.load(meta_path, map_location=DEVICE, weights_only=False)
    models = []
    for i in range(meta["n_models"]):
        f = model_path / f"model_{i}.pt"
        if not f.exists():
            print(f"Error: model file not found: {f}", file=sys.stderr)
            sys.exit(1)
        cp = torch.load(f, map_location=DEVICE, weights_only=False)
        m = DualHeadCNN(input_channels=11).to(DEVICE)
        m.load_state_dict(cp["model_state_dict"])
        m.eval()
        models.append(m)
    return models


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: data file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    measurements, is_control = [], []
    with open(filepath, "r") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.strip().split("\t")
            if len(parts) < N_MEASUREMENTS + 1:
                print(
                    f"Error: line {lineno} has {len(parts)} columns, expected at least {N_MEASUREMENTS + 1}",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                meas = [float(x) for x in parts[:N_MEASUREMENTS]]
            except ValueError:
                print(
                    f"Error: non-numeric measurement on line {lineno}", file=sys.stderr
                )
                sys.exit(1)
            try:
                ctrl = int(parts[N_MEASUREMENTS])
            except ValueError:
                print(
                    f"Error: non-integer is_control value on line {lineno}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if ctrl not in (0, 1, 2):
                print(
                    f"Error: invalid is_control={ctrl} on line {lineno} (expected 0, 1, or 2)",
                    file=sys.stderr,
                )
                sys.exit(1)
            measurements.append(meas)
            is_control.append(ctrl)

    if not measurements:
        print("Error: no data rows found", file=sys.stderr)
        sys.exit(1)

    meas_arr = np.array(measurements, dtype=np.float32)
    ctrl_arr = np.array(is_control, dtype=np.int32)

    mask_78 = ctrl_arr == 1
    mask_77 = ctrl_arr == 2
    mask_sample = ctrl_arr == 0

    if not np.any(mask_78):
        print("Error: no 7_8 control found (is_control=1)", file=sys.stderr)
        sys.exit(1)
    if not np.any(mask_77):
        print("Error: no 7_7 control found (is_control=2)", file=sys.stderr)
        sys.exit(1)
    if not np.any(mask_sample):
        print("Error: no samples to classify (all rows are controls)", file=sys.stderr)
        sys.exit(1)

    return (
        meas_arr[mask_sample],
        meas_arr[mask_78][0],
        meas_arr[mask_77][0],
        np.where(mask_sample)[0],
    )


def prepare(
    sample: np.ndarray, ctrl78: np.ndarray, ctrl77: np.ndarray, noise: float = 0.0
) -> torch.Tensor:
    s = sample.flatten().copy()
    c78 = ctrl78.flatten().copy()
    c77 = ctrl77.flatten().copy()
    if noise > 0:
        s += np.random.normal(0, noise * s.std(), s.shape)
        c78 += np.random.normal(0, noise * c78.std(), c78.shape)
        c77 += np.random.normal(0, noise * c77.std(), c77.shape)

    s_d1, s_d2 = compute_derivatives(s.reshape(1, -1))
    s_d1, s_d2 = s_d1.squeeze(), s_d2.squeeze()
    c78_d1, _ = compute_derivatives(c78.reshape(1, -1))
    c78_d1 = c78_d1.squeeze()
    c77_d1, _ = compute_derivatives(c77.reshape(1, -1))
    c77_d1 = c77_d1.squeeze()

    channels = np.stack(
        [
            normalize(s),
            normalize(s_d1),
            normalize(s_d2),
            normalize(s - c78),
            normalize(s_d1 - c78_d1),
            normalize(c78),
            normalize(c78_d1),
            normalize(s - c77),
            normalize(s_d1 - c77_d1),
            normalize(c77),
            normalize(c77_d1),
        ],
        axis=0,
    )
    return torch.tensor(channels, dtype=torch.float32).unsqueeze(0)


def predict(
    models: List[DualHeadCNN],
    sample: np.ndarray,
    ctrl78: np.ndarray,
    ctrl77: np.ndarray,
) -> Dict:
    all_p1, all_p2 = [], []
    for model in models:
        mp1, mp2 = [], []
        with torch.no_grad():
            x = prepare(sample, ctrl78, ctrl77).to(DEVICE)
            l1, l2 = model(x)
            mp1.append(torch.softmax(l1, dim=1))
            mp2.append(torch.softmax(l2, dim=1))
            for _ in range(TTA_RUNS - 1):
                x = prepare(sample, ctrl78, ctrl77, noise=0.01).to(DEVICE)
                l1, l2 = model(x)
                mp1.append(torch.softmax(l1, dim=1))
                mp2.append(torch.softmax(l2, dim=1))
        all_p1.append(torch.stack(mp1).mean(dim=0))
        all_p2.append(torch.stack(mp2).mean(dim=0))

    p1 = torch.stack(all_p1).mean(dim=0)
    p2 = torch.stack(all_p2).mean(dim=0)
    i1, i2 = p1.argmax(dim=1).item(), p2.argmax(dim=1).item()
    c1, c2 = p1[0, i1].item(), p2[0, i2].item()
    a1, a2 = IDX_TO_ALLELE[i1], IDX_TO_ALLELE[i2]

    if a1 == "K-" or a2 == "K-":
        genotype = "K-"
    else:
        n1, n2 = int(a1), int(a2)
        if n1 > n2:
            n1, n2 = n2, n1
            c1, c2 = c2, c1
        genotype = f"{n1}_{n2}"

    return {
        "genotype": genotype,
        "conf_allele1": c1,
        "conf_allele2": c2,
        "min_conf": min(c1, c2),
    }


def main():
    parser = argparse.ArgumentParser(description="PCR Genotype Inference")
    parser.add_argument("--data", "-d", required=True, help="Path to inference TSV")
    parser.add_argument(
        "--model", "-m", default="models", help="Model directory (default: models/)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output TSV path (optional)"
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: model directory not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    models = load_ensemble(args.model)
    samples, ctrl78, ctrl77, indices = load_data(args.data)

    results = []
    for i, idx in enumerate(indices):
        r = predict(models, samples[i], ctrl78, ctrl77)
        r["row"] = idx + 1
        results.append(r)

    for r in results:
        flag = "" if r["min_conf"] > 0.8 else " ⚠️" if r["min_conf"] > 0.65 else " ❌"
        print(
            f"Row {r['row']:3d}: {r['genotype']:5s}  (conf: {r['conf_allele1']:.2f}, {r['conf_allele2']:.2f}){flag}"
        )

    if args.output:
        with open(args.output, "w") as f:
            f.write("row\tgenotype\tconf_allele1\tconf_allele2\n")
            for r in results:
                f.write(
                    f"{r['row']}\t{r['genotype']}\t{r['conf_allele1']:.4f}\t{r['conf_allele2']:.4f}\n"
                )


if __name__ == "__main__":
    main()
