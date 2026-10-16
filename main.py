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


from config import (
    set_seed,
    DEVICE,
    RANDOM_SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    VALIDATION_SPLIT,
    IDX_TO_ALLELE,
)
from data import (
    PCRDataPipeline,
    load_inference_data,
    prepare_sample_for_inference,
    parse_genotype,
)
from model import DualHeadCNN, DualHeadLoss
from trainer import ModelTrainer


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


def train_ensemble(
    dataset_path: str,
    n_models: int = 3,
    base_seed: int = RANDOM_SEED,
    **train_kwargs,
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
        pipeline = PCRDataPipeline(
            filepath=dataset_path,
            batch_size=BATCH_SIZE,
            val_split=VALIDATION_SPLIT,
            seed=seed,
            augment_train=False,
        )
        train_loader, val_loader, train_genotypes, _, c78, c77 = (
            pipeline.create_data_loaders()
        )
        if i == 0:
            control_78_lookup, control_77_lookup = c78, c77
        model = DualHeadCNN(input_channels=11).to(DEVICE)
        temp_trainer = ModelTrainer(model, None, None, None, None, None, DEVICE)
        weights1, weights2 = temp_trainer.compute_class_weights(train_genotypes)
        criterion = DualHeadLoss(
            weights1=weights1.to(DEVICE),
            weights2=weights2.to(DEVICE),
            label_smoothing=0.05,
        )
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE * 3,
            epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )

        trainer = ModelTrainer(
            model, train_loader, val_loader, optimizer, scheduler, criterion, DEVICE
        )
        trainer.train_model(num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)

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

    conf1 = probs1[0, int(pred1)].item()
    conf2 = probs2[0, int(pred2)].item()

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

        conf1 = probs1[0, int(pred1)].item()
        conf2 = probs2[0, int(pred2)].item()

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


def run_inference(
    models: List[DualHeadCNN],
    data_path: str,
    output_path: Optional[str] = None,
    device: torch.device = DEVICE,
    seq_len: int = 46,
) -> List[Dict]:

    print(f"\nLoading inference data from: {data_path}")
    sample_meas, control_78_meas, control_77_meas, sample_indices = load_inference_data(
        data_path, seq_len=seq_len
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
        print(f"\n⚠️  {low_conf} samples with confidence < 0.9")

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
        description="PCR Melting Curve Genotype Classification"
    )
    parser.add_argument("mode", choices=["train", "infer"], help="Operation mode")
    parser.add_argument(
        "--data", "-d", type=str, default="dataset.tsv", help="Path to TSV data"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="models", help="Output directory"
    )

    parser.add_argument(
        "--ensemble",
        "-n",
        type=int,
        default=5,
        help="Number of models in ensemble (default: 5)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=46,
        help="Количество точек температуры (например, 45 или 46)",
    )

    return parser.parse_args()


def cmd_train(args):
    set_seed(RANDOM_SEED)
    dataset_path = Path(args.data)
    output_dir = Path(args.output)

    print("=" * 60)
    print("PCR Melting Curve Genotype Classification - Training")
    print(f"Dataset:  {dataset_path}")
    print(f"Ensemble: {args.ensemble} models")
    print("=" * 60)

    models, control_78_lookup, control_77_lookup = train_ensemble(
        dataset_path=str(dataset_path), n_models=args.ensemble, base_seed=RANDOM_SEED
    )

    save_ensemble(models, control_78_lookup, control_77_lookup, str(output_dir))

    print("\n" + "=" * 60)
    print("Validation on Training Dataset")
    print("=" * 60)

    pipeline = PCRDataPipeline(
        str(dataset_path),
        BATCH_SIZE,
        VALIDATION_SPLIT,
        RANDOM_SEED,
        False,
        seq_len=args.seq_len,
    )
    measurements, allele_pairs, raw_genotypes, run_ids, is_control = (
        pipeline.load_dataset()
    )

    unique_runs = np.unique(run_ids)
    np.random.seed(RANDOM_SEED)

    val_indices_set = set()
    for run_id in unique_runs:
        run_mask = (run_ids == run_id) & (is_control == 0)
        run_indices = np.where(run_mask)[0]
        if len(run_indices) == 0:
            continue
        run_genotypes_list = [raw_genotypes[i] for i in run_indices]
        try:
            _, val_idx = train_test_split(
                run_indices,
                test_size=VALIDATION_SPLIT,
                random_state=RANDOM_SEED,
                stratify=run_genotypes_list,
            )
        except ValueError:
            _, val_idx = train_test_split(
                run_indices, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
            )
        val_indices_set.update(val_idx)

    control_78_lookup_val = pipeline.build_control_lookup()
    control_77_lookup_val = {}
    for i in range(len(measurements)):
        rid = int(run_ids[i])
        if raw_genotypes[i] == "7_7" and rid not in control_77_lookup_val:
            control_77_lookup_val[rid] = measurements[i].copy()

    train_correct, train_total = 0, 0
    val_correct, val_total = 0, 0
    misclassified = []
    y_true_val, y_pred_val = [], []

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

    print("\n" + "=" * 70)
    print("FINAL METRICS OF THE ENSEMBLE")
    print("=" * 70)

    print(
        f"Train Acc:     {train_correct}/{train_total} ({100*train_correct/train_total:.2f}%)"
    )
    if val_total > 0:
        print(
            f"Val Acc:      {val_correct}/{val_total} ({100*val_correct/val_total:.2f}%)"
        )
        val_a1_correct = sum(
            1
            for t, p in zip(y_true_val, y_pred_val)
            if parse_genotype(t)[0] == parse_genotype(p)[0]
        )
        val_a2_correct = sum(
            1
            for t, p in zip(y_true_val, y_pred_val)
            if parse_genotype(t)[1] == parse_genotype(p)[1]
        )
        print(f"Acc for Allele 1:   {100*val_a1_correct/val_total:.2f}%")
        print(f"Acc for Allele 2:   {100*val_a2_correct/val_total:.2f}%")

        print("\n--- Classification Report ---")
        labels = sorted(list(set(y_true_val + y_pred_val)))
        print(
            classification_report(
                y_true_val, y_pred_val, labels=labels, zero_division=0
            )
        )

        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_true_val, y_pred_val, labels=labels)
        title = "True \ Pred"
        header = f"{title:>12} | " + " | ".join([f"{l:>5}" for l in labels])
        print(header)
        print("-" * len(header))
        for idx, true_label in enumerate(labels):
            row = f"{true_label:>12} | " + " | ".join([f"{val:>5}" for val in cm[idx]])
            print(row)

    print(f"\nTotal errors (training + validation): {len(misclassified)}")
    if misclassified:
        print("\nList of incorrectly classified samples")
        print("-" * 70)
        for m in misclassified:
            split_flag = "VAL" if m["is_val"] else "TRN"
            print(
                f"  [{split_flag}] Sample {m['index']:4d} | Run {m['run_id']:2d}: True={m['true']:5s}, Pred={m['pred']:5s} (conf: {m['conf'][0]:.2f}, {m['conf'][1]:.2f})"
            )
    else:
        print(f"\nNo errors! 100% accuracy.")

    return 0


def cmd_infer(args):
    set_seed(RANDOM_SEED)
    print(f"Starting inference using models from  {args.output}...")
    run_inference(
        models=models,
        data_path=args.data,
        output_path="prediction.tsv",
        device=DEVICE,
        seq_len=args.seq_len,
    )
    return 0


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        exit(cmd_train(args))
    elif args.mode == "infer":
        exit(cmd_infer(args))
