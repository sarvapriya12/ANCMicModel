"""Fine-tune FastEnhancer on stage-1 outputs of the synthetic battlefield set.

Same architecture as the shipped model (so the streaming ONNX export, its
compute cost and the Pi benchmark are unchanged), same loss *recipe* as the
upstream training config (compressed-magnitude MSE 0.3, compressed-complex MSE
0.2, STFT-consistency 0.3, waveform L1 0.2), a small learning rate and a short
schedule. The custom losses in ``losses.py`` are deliberately not used here: the
upstream recipe is what this model was trained with.

The upstream repo (MIT) provides the model and the compressed STFT; it is
imported from ``third_party/fastenhancer``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FinetuneConfig:
    data_dir: str
    out_dir: str
    checkpoint: str
    config_yaml: str
    upstream_root: str
    epochs: int = 6
    batch_size: int = 16
    crop_s: float = 3.0
    lr: float = 2.0e-4
    lr_min: float = 1.0e-5
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    gain_aug_db: float = 4.0
    seed: int = 0
    steps_per_epoch: int | None = None
    loss_weights: dict = field(
        default_factory=lambda: {
            "mag_mse": 0.3,
            "complex_mse": 0.2,
            "consistency": 0.3,
            "wav_l1": 0.2,
        }
    )


def _load_model(cfg: FinetuneConfig, device):
    import torch
    from omegaconf import OmegaConf

    root = str(Path(cfg.upstream_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from functional import CompressedSTFT  # type: ignore
    from models.fastenhancer.default.model import Model  # type: ignore

    kwargs = OmegaConf.to_container(
        OmegaConf.load(cfg.config_yaml).model_kwargs, resolve=True
    )
    model = Model(**kwargs)
    ck = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    cons_stft = CompressedSTFT(
        n_fft=512, hop_size=256, win_size=512, win_type="hann", compression=0.3
    ).to(device)
    return model, cons_stft


def _losses(model, cons_stft, noisy, clean, w):
    import torch
    import torch.nn.functional as F

    wav_hat, spec_hat = model(noisy)
    spec = model.stft(clean)
    n = min(wav_hat.shape[-1], clean.shape[-1])
    mag = F.mse_loss(
        torch.linalg.norm(spec_hat, dim=-1), torch.linalg.norm(spec, dim=-1)
    )
    cplx = F.mse_loss(spec_hat, spec)
    cons = F.mse_loss(cons_stft(wav_hat[..., :n]), cons_stft(clean[..., :n]))
    l1 = F.l1_loss(wav_hat[..., :n], clean[..., :n])
    total = (
        w["mag_mse"] * mag
        + w["complex_mse"] * cplx
        + w["consistency"] * cons
        + w["wav_l1"] * l1
    )
    parts = {
        "mag_mse": mag.item(),
        "complex_mse": cplx.item(),
        "consistency": cons.item(),
        "wav_l1": l1.item(),
    }
    return total, parts, wav_hat[..., :n]


def _si_sdr(est, ref, eps=1e-8):
    import torch

    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return 10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + eps) + eps)


def _lr_at(step, total, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * p))


def validate(model, cons_stft, vx, vy, cfg, device, max_items=240):
    import torch

    model.eval()
    tot, sis, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(vx), max_items), cfg.batch_size):
            x = torch.from_numpy(np.asarray(vx[i : i + cfg.batch_size], np.float32)).to(
                device
            )
            y = torch.from_numpy(np.asarray(vy[i : i + cfg.batch_size], np.float32)).to(
                device
            )
            loss, _, wav = _losses(model, cons_stft, x, y, cfg.loss_weights)
            tot += loss.item() * x.shape[0]
            sis += _si_sdr(wav, y[..., : wav.shape[-1]]).sum().item()
            n += x.shape[0]
    model.train()
    return tot / n, sis / n


def run_finetune(cfg: FinetuneConfig) -> dict:
    import torch

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = Path(cfg.data_dir)
    xs = np.load(data / "x.npy", mmap_mode="r")
    ys = np.load(data / "y.npy", mmap_mode="r")
    vx = np.load(data / "val_x.npy", mmap_mode="r")
    vy = np.load(data / "val_y.npy", mmap_mode="r")

    model, cons_stft = _load_model(cfg, device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    crop = int(cfg.crop_s * 48_000)
    spe = cfg.steps_per_epoch or max(len(xs) // cfg.batch_size, 1)
    total = cfg.epochs * spe

    base_loss, base_si = validate(model, cons_stft, vx, vy, cfg, device)
    print(
        f"baseline (stock weights): val loss {base_loss:.5f}  SI-SDR {base_si:.2f} dB",
        flush=True,
    )
    best = (base_loss, base_si, -1)
    log = [{"step": 0, "val_loss": base_loss, "val_si_sdr": base_si}]

    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        run = 0.0
        for _ in range(spe):
            idx = np.sort(rng.integers(0, len(xs), cfg.batch_size))
            starts = rng.integers(0, xs.shape[1] - crop + 1, cfg.batch_size)
            xb = np.stack([xs[i, s : s + crop] for i, s in zip(idx, starts)]).astype(
                np.float32
            )
            yb = np.stack([ys[i, s : s + crop] for i, s in zip(idx, starts)]).astype(
                np.float32
            )
            g = 10.0 ** (
                rng.uniform(-cfg.gain_aug_db, cfg.gain_aug_db, (cfg.batch_size, 1))
                / 20.0
            )
            xb *= g.astype(np.float32)
            yb *= g.astype(np.float32)
            x = torch.from_numpy(xb).to(device)
            y = torch.from_numpy(yb).to(device)

            for pg in opt.param_groups:
                pg["lr"] = _lr_at(step, total, cfg)
            loss, _parts, _ = _losses(model, cons_stft, x, y, cfg.loss_weights)
            if not torch.isfinite(loss):
                print("non-finite loss; skipping step", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run += loss.item()
            step += 1

        vl, vs = validate(model, cons_stft, vx, vy, cfg, device)
        log.append(
            {"step": step, "train_loss": run / spe, "val_loss": vl, "val_si_sdr": vs}
        )
        print(
            f"epoch {epoch + 1}/{cfg.epochs}  train {run / spe:.5f}  val {vl:.5f}  "
            f"SI-SDR {vs:.2f} dB  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if vl < best[0]:
            best = (vl, vs, epoch)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1}, out / "best.pth"
            )
        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, out / "last.pth")

    (out / "log.json").write_text(
        json.dumps(
            {"baseline": log[0], "history": log, "best_epoch": best[2] + 1}, indent=1
        )
    )
    return {
        "baseline_val_loss": base_loss,
        "baseline_val_si_sdr": base_si,
        "best_val_loss": best[0],
        "best_val_si_sdr": best[1],
        "best_epoch": best[2] + 1,
        "improved": best[2] >= 0,
        "best_path": str(out / "best.pth") if best[2] >= 0 else None,
    }
