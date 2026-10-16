"""
=============================================================================
  Autoregressive Language Modeling Benchmark System across Recurrent Architectures
  Student : Shah Karan Prakash
  Program : B.Tech in Computer Science and Engineering
  Dataset : Frankenstein; Or, The Modern Prometheus – Project Gutenberg eBook #84
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports & Global Seed
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import math
import random
import requests
import warnings
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Ingestion & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
FALLBACK_URL  = "https://www.gutenberg.org/ebooks/84.txt.utf-8"
LOCAL_CACHE   = "frankenstein_raw.txt"

VOCAB_SIZE_LIMIT = 6_000
SEQ_LEN          = 20
PAD_IDX          = 0
UNK_IDX          = 1

PUNCTUATION = {".", ",", ";", ":", "!", "?"}

def download_text(url: str, fallback: str) -> str:
    """Download raw Gutenberg text with a fallback mirror."""
    for u in [url, fallback]:
        try:
            print(f"[INFO] Fetching text from: {u}")
            resp = requests.get(u, timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            print(f"[WARN] {u} → {exc}")
    raise RuntimeError("Both download URLs failed.")

def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    # Header ends after the line starting with "*** START OF"
    start_marker = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG", text, re.IGNORECASE)
    if start_marker:
        text = text[start_marker.end():]

    # Footer begins at "*** END OF"
    end_marker = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG", text, re.IGNORECASE)
    if end_marker:
        text = text[:end_marker.start()]

    return text.strip()

def tokenize(text: str) -> list:
    """Lowercase, isolate punctuation, split on whitespace."""
    text = text.lower()
    # Insert spaces around target punctuation so they become standalone tokens
    for ch in PUNCTUATION:
        text = text.replace(ch, f" {ch} ")
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def build_vocab(tokens: list, max_vocab: int) -> tuple:
    """Return (word2idx, idx2word) with <pad>=0, <unk>=1."""
    counter = Counter(tokens)
    most_common = counter.most_common(max_vocab)
    # idx 0 → <pad>, idx 1 → <unk>
    word2idx = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for word, _ in most_common:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word

def encode(tokens: list, word2idx: dict) -> list:
    return [word2idx.get(t, UNK_IDX) for t in tokens]

def load_data():
    """Full ingestion pipeline – returns encoded token list + vocab dicts."""
    if os.path.exists(LOCAL_CACHE):
        print("[INFO] Loading text from local cache.")
        with open(LOCAL_CACHE, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = download_text(GUTENBERG_URL, FALLBACK_URL)
        with open(LOCAL_CACHE, "w", encoding="utf-8") as f:
            f.write(raw)

    text   = strip_gutenberg(raw)
    tokens = tokenize(text)
    print(f"[INFO] Total tokens after preprocessing: {len(tokens):,}")

    word2idx, idx2word = build_vocab(tokens, VOCAB_SIZE_LIMIT)
    print(f"[INFO] Vocabulary size: {len(word2idx):,}")

    encoded = encode(tokens, word2idx)
    return encoded, word2idx, idx2word


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, sequences: list, targets: list):
        self.X = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(targets,   dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_dataloaders(encoded: list, seq_len: int, batch_size: int):
    """Sliding-window sequences; chronological 80/10/10 split."""
    sequences, targets = [], []
    for i in range(len(encoded) - seq_len):
        sequences.append(encoded[i : i + seq_len])
        targets.append(encoded[i + seq_len])

    n      = len(sequences)
    n_tr   = int(0.80 * n)
    n_val  = int(0.10 * n)
    # remainder → test

    tr_seqs,  tr_tgts  = sequences[:n_tr],             targets[:n_tr]
    val_seqs, val_tgts = sequences[n_tr:n_tr+n_val],   targets[n_tr:n_tr+n_val]
    te_seqs,  te_tgts  = sequences[n_tr+n_val:],        targets[n_tr+n_val:]

    print(f"[INFO] Train samples: {len(tr_seqs):,} | Val: {len(val_seqs):,} | Test: {len(te_seqs):,}")

    train_dl = DataLoader(TextDataset(tr_seqs,  tr_tgts),  batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(TextDataset(val_seqs, val_tgts), batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl  = DataLoader(TextDataset(te_seqs,  te_tgts),  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
class AutoregressiveLM(nn.Module):
    """
    Unified language-model class supporting:
      - 'rnn'   : Simple (Elman) RNN
      - 'birnn' : Bidirectional RNN
      - 'lstm'  : Long Short-Term Memory
      - 'gru'   : Gated Recurrent Unit
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 arch: str, dropout: float = 0.20):
        super().__init__()
        self.arch       = arch.lower()
        self.hidden_dim = hidden_dim
        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.drop       = nn.Dropout(dropout)

        bidir = (self.arch == "birnn")
        rnn_type = "rnn" if self.arch in ("rnn", "birnn") else self.arch

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = 1,
            batch_first   = True,
            dropout       = 0.0,      # single layer → PyTorch ignores anyway
            bidirectional = bidir,
        )

        rnn_out_dim = hidden_dim * (2 if bidir else 1)
        self.fc = nn.Linear(rnn_out_dim, vocab_size)

    def forward(self, x):
        emb = self.drop(self.embed(x))        # (B, L, E)
        out, _ = self.rnn(emb)                # (B, L, H*dirs)
        logits  = self.fc(out[:, -1, :])      # last time-step → (B, V)
        return logits

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training & Evaluation Utilities
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, clip=5.0):
    """One pass through the data. If optimizer is None, run in eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_samples = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            total_loss    += loss.item() * y.size(0)
            total_samples += y.size(0)

    return total_loss / total_samples


def evaluate_test(model, loader, criterion):
    """Return loss, PPL, top-1 accuracy (%), top-5 accuracy (%)."""
    model.eval()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)                        # (B, V)
            loss   = criterion(logits, y)

            total_loss += loss.item() * y.size(0)

            _, top5 = logits.topk(5, dim=1)          # (B, 5)
            correct1 += (top5[:, 0] == y).sum().item()
            correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total    += y.size(0)

    avg_loss = total_loss / total
    ppl      = math.exp(min(avg_loss, 20))           # cap to avoid overflow
    acc1     = 100.0 * correct1 / total
    acc5     = 100.0 * correct5 / total
    return avg_loss, ppl, acc1, acc5


# ─────────────────────────────────────────────────────────────────────────────
# 5. Text Generation with Temperature Sampling
# ─────────────────────────────────────────────────────────────────────────────
def generate_text(model, prompt: str, word2idx: dict, idx2word: dict,
                  seq_len: int, num_words: int = 50, temperature: float = 1.0) -> str:
    model.eval()
    tokens = tokenize(prompt)
    ids    = encode(tokens, word2idx)

    # Pad or trim to seq_len
    if len(ids) < seq_len:
        ids = [PAD_IDX] * (seq_len - len(ids)) + ids
    else:
        ids = ids[-seq_len:]

    generated = list(tokens)
    with torch.no_grad():
        for _ in range(num_words):
            x = torch.tensor([ids], dtype=torch.long).to(DEVICE)   # (1, L)
            logits = model(x)[0]                                    # (V,)
            logits /= max(temperature, 1e-8)
            probs   = torch.softmax(logits, dim=-1).cpu().numpy()
            next_id = int(np.random.choice(len(probs), p=probs))
            generated.append(idx2word.get(next_id, "<unk>"))
            ids = ids[1:] + [next_id]

    # Merge punctuation tokens
    out = []
    for tok in generated:
        if tok in PUNCTUATION and out:
            out[-1] += tok
        else:
            out.append(tok)
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Lexical Diversity Metrics
# ─────────────────────────────────────────────────────────────────────────────
def distinct_n(text: str, n: int) -> float:
    words  = text.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Benchmark Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Hyperparameters ─────────────────────────────────────────────────────
    EMBED_DIM  = 128
    HIDDEN_DIM = 256
    DROPOUT    = 0.20
    LR         = 1e-3
    EPOCHS     = 8
    BATCH_SIZE = 128
    CLIP       = 5.0

    ARCHS = ["rnn", "birnn", "lstm", "gru"]
    ARCH_LABELS = {
        "rnn":   "Simple RNN",
        "birnn": "Bidirectional RNN",
        "lstm":  "LSTM",
        "gru":   "GRU",
    }

    PROMPTS = [
        "i stood beside the desk and saw",
        "the stormy night was near its close",
        "a silent flash of light illuminated",
    ]
    TEMPERATURES = [0.5, 1.0, 1.5]

    # ── Data ────────────────────────────────────────────────────────────────
    encoded, word2idx, idx2word = load_data()
    vocab_size = len(word2idx)
    train_dl, val_dl, test_dl = make_dataloaders(encoded, SEQ_LEN, BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()

    # Storage for results
    all_train_losses = {}   # arch → [epoch_loss, ...]
    all_val_losses   = {}
    results          = {}

    # ── Training ─────────────────────────────────────────────────────────────
    trained_models = {}
    for arch in ARCHS:
        label = ARCH_LABELS[arch]
        print(f"\n{'='*60}")
        print(f"  Training : {label}")
        print(f"{'='*60}")

        model = AutoregressiveLM(
            vocab_size = vocab_size,
            embed_dim  = EMBED_DIM,
            hidden_dim = HIDDEN_DIM,
            arch       = arch,
            dropout    = DROPOUT,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        params    = count_params(model)
        print(f"  Params   : {params:,}")

        train_losses, val_losses = [], []
        t_start = time.time()

        for epoch in range(1, EPOCHS + 1):
            tr_loss  = run_epoch(model, train_dl, criterion, optimizer, CLIP)
            val_loss = run_epoch(model, val_dl,   criterion, optimizer=None)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            print(f"  Epoch {epoch:02d}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        runtime = time.time() - t_start
        print(f"  Runtime  : {runtime:.1f}s")

        # ── Test Evaluation ──────────────────────────────────────────────────
        test_loss, ppl, acc1, acc5 = evaluate_test(model, test_dl, criterion)
        print(f"  Test Loss: {test_loss:.4f} | PPL: {ppl:.2f} | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")

        all_train_losses[arch] = train_losses
        all_val_losses[arch]   = val_losses
        trained_models[arch]   = model

        results[arch] = {
            "label"      : label,
            "params"     : params,
            "runtime_s"  : round(runtime, 2),
            "train_losses": train_losses,
            "val_losses"  : val_losses,
            "test_loss"  : round(test_loss, 4),
            "perplexity" : round(ppl, 2),
            "top1_acc"   : round(acc1, 2),
            "top5_acc"   : round(acc5, 2),
        }

    # ── Loss Comparison Plot ─────────────────────────────────────────────────
    print("\n[INFO] Generating loss_comparison_curves.png …")
    COLORS = {
        "rnn":   "#E76F51",
        "birnn": "#2A9D8F",
        "lstm":  "#457B9D",
        "gru":   "#F4A261",
    }

    fig = plt.figure(figsize=(16, 7), facecolor="#0F1117")
    fig.suptitle(
        "Training vs Validation Loss – All Four Recurrent Architectures",
        fontsize=16, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_tr  = fig.add_subplot(gs[0])
    ax_val = fig.add_subplot(gs[1])

    epochs_x = list(range(1, EPOCHS + 1))
    for arch in ARCHS:
        c = COLORS[arch]; lbl = ARCH_LABELS[arch]
        ax_tr.plot(epochs_x,  all_train_losses[arch], color=c, marker="o", lw=2.2, label=lbl)
        ax_val.plot(epochs_x, all_val_losses[arch],   color=c, marker="s", lw=2.2, label=lbl, linestyle="--")

    for ax, title in [(ax_tr, "Training Loss"), (ax_val, "Validation Loss")]:
        ax.set_facecolor("#1A1D27")
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.set_xlabel("Epoch", color="#AAAAAA", fontsize=11)
        ax.set_ylabel("Cross-Entropy Loss", color="#AAAAAA", fontsize=11)
        ax.tick_params(colors="#BBBBBB", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(True, linestyle="--", alpha=0.25, color="#666666")
        ax.legend(fontsize=9, framealpha=0.3, facecolor="#1A1D27",
                  labelcolor="white", edgecolor="#555555")

    plt.tight_layout()
    plt.savefig("loss_comparison_curves.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("[INFO] Saved loss_comparison_curves.png")

    # ── Text Generation & Lexical Diversity ─────────────────────────────────
    print("\n[INFO] Running text generation …")
    generation_results = {}

    for arch in ARCHS:
        model = trained_models[arch]
        generation_results[arch] = {}
        for temp in TEMPERATURES:
            temp_key = f"T={temp}"
            samples  = []
            d1_list, d2_list = [], []
            for prompt in PROMPTS:
                text = generate_text(model, prompt, word2idx, idx2word,
                                     SEQ_LEN, num_words=60, temperature=temp)
                samples.append(text)
                d1_list.append(distinct_n(text, 1))
                d2_list.append(distinct_n(text, 2))

            generation_results[arch][temp_key] = {
                "samples"    : samples,
                "distinct_1" : round(float(np.mean(d1_list)), 4),
                "distinct_2" : round(float(np.mean(d2_list)), 4),
            }

    # ── Print Generation Samples ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TEXT GENERATION SAMPLES")
    print("="*70)
    for arch in ARCHS:
        print(f"\n  ── {ARCH_LABELS[arch]} ──")
        for temp in TEMPERATURES:
            temp_key = f"T={temp}"
            d1 = generation_results[arch][temp_key]["distinct_1"]
            d2 = generation_results[arch][temp_key]["distinct_2"]
            sample = generation_results[arch][temp_key]["samples"][0]
            print(f"\n  Temperature {temp} | Distinct-1: {d1:.4f} | Distinct-2: {d2:.4f}")
            print(f"  Prompt : '{PROMPTS[0]}'")
            print(f"  Output : {sample[:180]}")

    # ── Results Summary ──────────────────────────────────────────────────────
    summary = {
        "metadata": {
            "student": "Shah Karan Prakash",
            "program": "B.Tech in Computer Science and Engineering",
            "dataset": "Frankenstein; Or, The Modern Prometheus – Project Gutenberg eBook #84",
            "device" : str(DEVICE),
            "vocab_size": vocab_size,
            "seq_len"   : SEQ_LEN,
            "embed_dim" : EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "epochs"    : EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        "benchmark": results,
        "generation": generation_results,
    }

    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[INFO] Saved results_summary.json")

    # ── Final Benchmark Table ────────────────────────────────────────────────
    print("\n" + "="*85)
    print(f"  {'Architecture':<22} {'Params':>10} {'Runtime(s)':>12} {'TestLoss':>10} {'PPL':>8} {'Top-1%':>8} {'Top-5%':>8}")
    print("="*85)
    for arch in ARCHS:
        r = results[arch]
        print(f"  {r['label']:<22} {r['params']:>10,} {r['runtime_s']:>12.1f} {r['test_loss']:>10.4f} {r['perplexity']:>8.2f} {r['top1_acc']:>8.2f} {r['top5_acc']:>8.2f}")
    print("="*85)
    print("\n[DONE] Benchmark pipeline complete.")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = main()
