import argparse
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml

from dataset import create_dataloaders
from metrics import compute_all_metrics
from models import build_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(params, cfg):
    opt = cfg["training"]["optimizer"].lower()
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt}")


def make_scheduler(optimizer, cfg):
    sched = cfg["training"]["scheduler"]
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"])
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["training"]["step_size"],
            gamma=cfg["training"]["step_gamma"])
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, k):
    model.eval()
    running_loss = 0.0
    all_logits, all_targets = [], []
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        running_loss += criterion(logits, targets).item() * imgs.size(0)
        all_logits.append(logits)
        all_targets.append(targets)

    metrics = compute_all_metrics(torch.cat(all_logits), torch.cat(all_targets), k)
    metrics["validation_loss"] = running_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    k = cfg["evaluation"]["top_k"]

    train_loader, val_loader = create_dataloaders(cfg)

    model = build_model(
        cfg["model"]["type"], cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = make_optimizer(model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    run_name = cfg["mlflow"]["run_name"] or f"{cfg['model']['type']}_lr{cfg['training']['learning_rate']}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type": cfg["model"]["type"],
            "pretrained": cfg["model"]["pretrained"],
            "num_classes": cfg["model"]["num_classes"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["epochs"],
            "optimizer": cfg["training"]["optimizer"],
            "weight_decay": cfg["training"]["weight_decay"],
            "scheduler": cfg["training"]["scheduler"],
            "image_size": cfg["data"]["image_size"],
            "top_k": k,
            "seed": cfg["seed"],
        })

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        mlflow.log_params({"device": str(device), "gpu_name": gpu_name})

        t_start = time.time()
        best_pk = 0.0

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val = evaluate(model, val_loader, criterion, device, k)
            epoch_time = time.time() - t0

            if scheduler:
                scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "validation_loss": val["validation_loss"],
                f"precision@{k}": val[f"precision@{k}"],
                f"recall@{k}": val[f"recall@{k}"],
                f"f1@{k}": val[f"f1@{k}"],
                "time_per_epoch": epoch_time,
            }, step=epoch)

            pk = val[f"precision@{k}"]
            print(
                f"[{epoch}/{cfg['training']['epochs']}] "
                f"train_loss={train_loss:.4f}  val_loss={val['validation_loss']:.4f}  "
                f"P@{k}={pk:.4f}  R@{k}={val[f'recall@{k}']:.4f}  "
                f"F1@{k}={val[f'f1@{k}']:.4f}  ({epoch_time:.1f}s)"
            )

            if pk > best_pk:
                best_pk = pk
                ckpt = Path("outputs") / run_name / "best_model.pth"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)
                mlflow.log_artifact(str(ckpt))

        total_time = time.time() - t_start
        mlflow.log_metric("total_training_time", total_time)
        mlflow.log_artifact(args.config)

        print(f"\nDone in {total_time:.1f}s, best P@{k}={best_pk:.4f}")


if __name__ == "__main__":
    main()
