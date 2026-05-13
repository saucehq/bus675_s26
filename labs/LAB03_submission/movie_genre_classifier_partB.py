"""
Lab 3 — Part B: ResNet18 Drop-In Image Branch
==============================================
Copy your completed movie_genre_classifier.py into this file, then
replace the ImageBranch class with the ResNet18 version below.
Everything else (TabularBranch, FusionHead, Dataset, training) should
remain identical to Part A.
"""

# YOUR CODE HERE
# Paste your completed Part A code here, then replace ImageBranch
# with the class below. No other changes should be needed.


# =============================================================================
# PROVIDED: ResNet18 ImageBranch (replaces your Part A ImageBranch)
# =============================================================================

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# =============================================================================
# Constants  (identical to Part A)
# =============================================================================

GENRES = ["Animation", "Comedy", "Documentary", "Horror", "Romance", "Sci-Fi"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

NUMERIC_COLS = ["runtime", "vote_average", "vote_count",
                "release_year", "popularity", "budget", "revenue"]

LIST_FIELDS = ["cast", "directors", "writers", "production_companies"]
SINGLE_CAT_FIELDS = ["mpaa_rating"]

IMAGE_SIZE   = 128
MAX_LIST_LEN = 20
TOP_N_VOCAB  = 50
EMBED_DIM    = 32


# =============================================================================
# PROVIDED: VocabBuilder  (unchanged)
# =============================================================================

class VocabBuilder:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, top_n=TOP_N_VOCAB):
        self.top_n  = top_n
        self.vocabs = {}
        self.sizes  = {}

    def fit(self, df):
        for field in LIST_FIELDS:
            if field not in df.columns:
                continue
            counts = Counter()
            for val in df[field].dropna():
                if val:
                    counts.update(v.strip() for v in str(val).split("|") if v.strip())
            top_tokens = [tok for tok, _ in counts.most_common(self.top_n)]
            vocab = {tok: idx + 2 for idx, tok in enumerate(top_tokens)}
            self.vocabs[field] = vocab
            self.sizes[field]  = len(vocab) + 2

        for field in SINGLE_CAT_FIELDS:
            if field not in df.columns:
                continue
            unique_vals = [v for v in df[field].unique()
                           if isinstance(v, str) and v.strip()]
            vocab = {v: idx + 2 for idx, v in enumerate(sorted(unique_vals))}
            self.vocabs[field] = vocab
            self.sizes[field]  = len(vocab) + 2
        return self

    def encode_list(self, val, field, max_len=MAX_LIST_LEN):
        vocab = self.vocabs.get(field, {})
        if not isinstance(val, str) or not val.strip():
            return [self.PAD_IDX] * max_len
        tokens = [v.strip() for v in val.split("|") if v.strip()]
        ids = [vocab.get(tok, self.UNK_IDX) for tok in tokens]
        ids = ids[:max_len]
        ids += [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def encode_single(self, val, field):
        vocab = self.vocabs.get(field, {})
        if not isinstance(val, str) or not val.strip():
            return self.PAD_IDX
        return vocab.get(val.strip(), self.UNK_IDX)

    def save(self, path):
        data = {"vocabs": self.vocabs, "sizes": self.sizes, "top_n": self.top_n}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        vb = cls(top_n=data["top_n"])
        vb.vocabs = data["vocabs"]
        vb.sizes  = data["sizes"]
        return vb


# =============================================================================
# PROVIDED: NumericScaler  (unchanged)
# =============================================================================

class NumericScaler:
    def __init__(self):
        self.means = {}
        self.stds  = {}

    def fit(self, df):
        for col in NUMERIC_COLS:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                self.means[col] = float(vals.mean())
                self.stds[col]  = max(float(vals.std()), 1e-8)
        return self

    def transform(self, df):
        result = {}
        for col in NUMERIC_COLS:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns \
                   else pd.Series([float("nan")] * len(df))
            vals = vals.fillna(self.means.get(col, 0.0))
            mean = self.means.get(col, 0.0)
            std  = self.stds.get(col, 1.0)
            result[col] = ((vals - mean) / std).values.astype(np.float32)
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({"means": self.means, "stds": self.stds}))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        ns = cls()
        ns.means = data["means"]
        ns.stds  = data["stds"]
        return ns


# =============================================================================
# Dataset  (unchanged from Part A)
# =============================================================================

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class MoviePosterDataset(Dataset):
    def __init__(self, df, image_dir, vocab_builder, numeric_scaler,
                 transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.vocab     = vocab_builder
        self.transform = transform if transform is not None else EVAL_TRANSFORM

        scaled = numeric_scaler.transform(df)
        self.numeric_matrix = np.stack(
            [scaled[col] for col in NUMERIC_COLS], axis=1
        ).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.image_dir / row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
        image = self.transform(img)

        numeric = torch.tensor(self.numeric_matrix[idx], dtype=torch.float32)

        cat_fields = {}
        for field in LIST_FIELDS:
            ids = self.vocab.encode_list(
                row.get(field, ""), field, max_len=MAX_LIST_LEN
            )
            cat_fields[field] = torch.tensor(ids, dtype=torch.long)

        cat_fields["mpaa_rating"] = torch.tensor(
            self.vocab.encode_single(row.get("mpaa_rating", ""), "mpaa_rating"),
            dtype=torch.long,
        )

        label = torch.tensor(GENRE_TO_IDX[row["label"]], dtype=torch.long)

        return {
            "image":      image,
            "numeric":    numeric,
            "cat_fields": cat_fields,
            "label":      label,
        }


# =============================================================================
# PROVIDED: ResNet18 ImageBranch  (replaces Part A custom CNN)
# =============================================================================

class ImageBranch(nn.Module):
    """
    Transfer learning image encoder: pretrained ResNet18 backbone
    with a small trainable projection head.

    The entire backbone is frozen by default (only the head trains).
    Optionally, the last residual block (layer4) can be unfrozen for
    fine-tuning once the head has converged.
    """

    BACKBONE_OUT_DIM = 512  # ResNet18 feature dimension after global average pool

    def __init__(self, out_dim=256, dropout=0.4, fine_tune=False):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in backbone.parameters():
            param.requires_grad = False

        if fine_tune:
            for param in backbone.layer4.parameters():
                param.requires_grad = True

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output: (batch, 512, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.BACKBONE_OUT_DIM, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = self.backbone(x)   # (batch, 512, 1, 1) — frozen
        return self.head(features)    # (batch, out_dim)   — trained


# =============================================================================
# TabularBranch  (unchanged from Part A)
# =============================================================================

class TabularBranch(nn.Module):
    def __init__(self, vocab_sizes, out_dim=256, dropout=0.3):
        super().__init__()

        n_numeric = len(NUMERIC_COLS)
        self.numeric_fc = nn.Sequential(
            nn.Linear(n_numeric, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.embeddings = nn.ModuleDict()
        total_embed_out = 0
        for field in LIST_FIELDS + SINGLE_CAT_FIELDS:
            vocab_size = vocab_sizes.get(field, 2)
            self.embeddings[field] = nn.Embedding(
                vocab_size, EMBED_DIM, padding_idx=VocabBuilder.PAD_IDX
            )
            total_embed_out += EMBED_DIM

        self.embed_fc = nn.Sequential(
            nn.Linear(total_embed_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.merge = nn.Sequential(
            nn.Linear(128 + 128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, numeric, cat_fields):
        num_out = self.numeric_fc(numeric)

        pooled = []
        for field in LIST_FIELDS:
            ids = cat_fields[field]
            emb = self.embeddings[field](ids)
            mask   = (ids != VocabBuilder.PAD_IDX).float().unsqueeze(-1)
            denom  = mask.sum(dim=1).clamp(min=1.0)
            pooled.append((emb * mask).sum(dim=1) / denom)

        mpaa_emb = self.embeddings["mpaa_rating"](cat_fields["mpaa_rating"])
        pooled.append(mpaa_emb)

        embed_cat = torch.cat(pooled, dim=1)
        emb_out   = self.embed_fc(embed_cat)

        merged = torch.cat([num_out, emb_out], dim=1)
        return self.merge(merged)


# =============================================================================
# FusionHead  (unchanged from Part A)
# =============================================================================

class FusionHead(nn.Module):
    def __init__(self, image_dim, tabular_dim, num_classes=len(GENRES), dropout=0.4):
        super().__init__()
        fused_dim = image_dim + tabular_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image_features, tabular_features):
        fused = torch.cat([image_features, tabular_features], dim=1)
        return self.classifier(fused)


# =============================================================================
# Full Model  (unchanged from Part A — drop-in just swaps ImageBranch)
# =============================================================================

class MultimodalGenreClassifier(nn.Module):
    def __init__(self, vocab_sizes, image_dim=256, tabular_dim=256):
        super().__init__()
        self.image_branch   = ImageBranch(out_dim=image_dim)
        self.tabular_branch = TabularBranch(vocab_sizes, out_dim=tabular_dim)
        self.fusion_head    = FusionHead(image_dim, tabular_dim)

    def forward(self, image, numeric, cat_fields):
        img_feat = self.image_branch(image)
        tab_feat = self.tabular_branch(numeric, cat_fields)
        return self.fusion_head(img_feat, tab_feat)


# =============================================================================
# Training utilities  (unchanged from Part A)
# =============================================================================

def collate_fn(batch):
    images   = torch.stack([s["image"]   for s in batch])
    numerics = torch.stack([s["numeric"] for s in batch])
    labels   = torch.stack([s["label"]   for s in batch])
    cat_fields = {}
    for key in batch[0]["cat_fields"]:
        cat_fields[key] = torch.stack([s["cat_fields"][key] for s in batch])
    return {"image": images, "numeric": numerics,
            "cat_fields": cat_fields, "label": labels}


def move_to_device(batch, device):
    return {
        "image":      batch["image"].to(device),
        "numeric":    batch["numeric"].to(device),
        "cat_fields": {k: v.to(device) for k, v in batch["cat_fields"].items()},
        "label":      batch["label"].to(device),
    }


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            batch  = move_to_device(batch, device)
            logits = model(batch["image"], batch["numeric"], batch["cat_fields"])
            loss   = criterion(logits, batch["label"])

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch["label"])
            preds       = logits.argmax(dim=1)
            correct    += (preds == batch["label"]).sum().item()
            total      += len(batch["label"])

    return total_loss / total, correct / total


def per_class_accuracy(model, loader, device):
    model.eval()
    counts  = {g: 0 for g in GENRES}
    correct = {g: 0 for g in GENRES}
    with torch.no_grad():
        for batch in loader:
            batch  = move_to_device(batch, device)
            logits = model(batch["image"], batch["numeric"], batch["cat_fields"])
            preds  = logits.argmax(dim=1)
            for pred, true in zip(preds.cpu(), batch["label"].cpu()):
                genre = GENRES[true.item()]
                counts[genre]  += 1
                correct[genre] += int(pred == true)
    return {g: correct[g] / counts[g] if counts[g] else 0.0 for g in GENRES}


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":              val_acc,
    }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train multimodal genre classifier (Part B)")
    parser.add_argument("--data_dir",       default="../data/movie_posters")
    parser.add_argument("--epochs",         type=int,   default=15)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-3)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints_partB")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    image_dir = data_dir / "images"
    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(data_dir / "train_manifest.csv")
    val_df   = pd.read_csv(data_dir / "val_manifest.csv")
    test_df  = pd.read_csv(data_dir / "test_manifest.csv")
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    vocab  = VocabBuilder(top_n=TOP_N_VOCAB).fit(train_df)
    scaler = NumericScaler().fit(train_df)
    vocab.save(ckpt_dir  / "vocab.json")
    scaler.save(ckpt_dir / "scaler.json")

    train_ds = MoviePosterDataset(train_df, image_dir, vocab, scaler,
                                  transform=TRAIN_TRANSFORM)
    val_ds   = MoviePosterDataset(val_df,   image_dir, vocab, scaler,
                                  transform=EVAL_TRANSFORM)
    test_ds  = MoviePosterDataset(test_df,  image_dir, vocab, scaler,
                                  transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    model     = MultimodalGenreClassifier(vocab_sizes=vocab.sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Report trainable vs frozen parameter counts (key Part B diagnostic)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")

    best_val_acc   = 0.0
    best_ckpt_path = ckpt_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion,
                                          optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt_path)
            print(f"  ✓ New best val acc={val_acc:.3f} — checkpoint saved")

    print("\nLoading best checkpoint for test evaluation …")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc = run_epoch(model, test_loader, criterion, optimizer,
                            device, train=False)
    per_class   = per_class_accuracy(model, test_loader, device)

    print(f"\nTest accuracy (overall): {test_acc:.3f}")
    print("\nPer-class accuracy:")
    print(f"  {'Genre':<15} {'Accuracy':>10}")
    print("  " + "-" * 27)
    for genre in GENRES:
        print(f"  {genre:<15} {per_class[genre]:>10.3f}")


if __name__ == "__main__":
    main()
