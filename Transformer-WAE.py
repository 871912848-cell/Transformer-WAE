import os
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def make_timestamp_dir(base_dir: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_spectra_from_excel(
    excel_path: str,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    df = pd.read_excel(excel_path)

    spec_df = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    starch_series = pd.to_numeric(df.iloc[:, -1], errors="coerce")

    spectra = spec_df.values.astype(np.float32)
    starch = starch_series.values.astype(np.float32).reshape(-1, 1)

    mask_valid = (~np.isnan(spectra).any(axis=1)) & (~np.isnan(starch).any(axis=1))
    if mask_valid.sum() < len(spectra):
        print(f"Dropping {len(spectra) - int(mask_valid.sum())} rows with NaNs.")

    spectra = spectra[mask_valid]
    starch = starch[mask_valid]

    spec_min = float(np.nanmin(spectra))
    spec_max = float(np.nanmax(spectra))

    spectra_01 = (spectra - spec_min) / (spec_max - spec_min + 1e-8)
    spectra_tanh = spectra_01 * 2.0 - 1.0

    starch_min = float(np.nanmin(starch))
    starch_max = float(np.nanmax(starch))

    X = torch.from_numpy(spectra_tanh).to(device)
    y = torch.from_numpy(starch).to(device)

    print(f"Loaded: {excel_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Spectra(mapped) range: {X.min().item():.3f} ~ {X.max().item():.3f}")
    print(f"Starch range: {y.min().item():.3f} ~ {y.max().item():.3f}")

    return X, y, spec_min, spec_max, starch_min, starch_max


class JointDataset(Dataset):
    def __init__(self, XY_norm: torch.Tensor):
        self.XY = XY_norm

    def __len__(self) -> int:
        return self.XY.shape[0]

    def __getitem__(self, idx: int):
        return self.XY[idx]


class SelfAttention1D(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "d_model must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, need_attn: bool = False):
        B, L, C = x.shape
        qkv = self.to_qkv(x).view(B, L, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return (out, attn) if need_attn else (out, None)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransformerBlock1DWithAttn(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttention1D(d_model, heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.last_attn = None

    def forward(self, src: torch.Tensor, need_attn: bool = False) -> torch.Tensor:
        x_norm = self.norm1(src)
        attn_out, attn_weights = self.self_attn(x_norm, need_attn=need_attn)
        src2 = src + self.dropout1(attn_out)

        x = self.norm2(src2)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        out = src2 + self.dropout2(x)

        self.last_attn = attn_weights.detach() if need_attn else None
        return out


class Encoder(nn.Module):
    def __init__(self, seq_len_total: int, latent_dim: int, d_model: int = 96, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.fc_in = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding1D(d_model, max_len=seq_len_total)
        self.blocks = nn.ModuleList([
            TransformerBlock1DWithAttn(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fc_z = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))

    def forward(self, x: torch.Tensor, attn_layer_idx: Optional[int] = None) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.fc_in(x)
        x = self.pos_enc(x)
        for i, blk in enumerate(self.blocks):
            need_attn = (attn_layer_idx is not None and i == attn_layer_idx)
            x = blk(x, need_attn=need_attn)
        x_mean = x.mean(dim=1)
        z = self.fc_z(x_mean)
        return z


class Decoder(nn.Module):
    def __init__(self, seq_len_total: int, latent_dim: int, d_model: int = 96, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.seq_len = seq_len_total
        self.d_model = d_model
        self.fc_in = nn.Linear(latent_dim, seq_len_total * d_model)
        self.pos_enc = PositionalEncoding1D(d_model, max_len=seq_len_total)
        self.blocks = nn.ModuleList([
            TransformerBlock1DWithAttn(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor, attn_layer_idx: Optional[int] = None) -> torch.Tensor:
        B = z.size(0)
        x = self.fc_in(z).view(B, self.seq_len, self.d_model)
        x = self.pos_enc(x)
        for i, blk in enumerate(self.blocks):
            need_attn = (attn_layer_idx is not None and i == attn_layer_idx)
            x = blk(x, need_attn=need_attn)
        x = self.fc_out(x).squeeze(-1)
        return torch.tanh(x)


class TransformerWAE(nn.Module):
    def __init__(self, seq_len_total: int, latent_dim: int, d_model: int = 96, nhead: int = 4,
                 enc_layers: int = 2, dec_layers: int = 3):
        super().__init__()
        self.encoder = Encoder(seq_len_total, latent_dim, d_model=d_model, nhead=nhead, num_layers=enc_layers)
        self.decoder = Decoder(seq_len_total, latent_dim, d_model=d_model, nhead=nhead, num_layers=dec_layers)

    def encode(self, x: torch.Tensor, attn_layer_idx: Optional[int] = None) -> torch.Tensor:
        return self.encoder(x, attn_layer_idx=attn_layer_idx)

    def decode(self, z: torch.Tensor, attn_layer_idx: Optional[int] = None) -> torch.Tensor:
        return self.decoder(z, attn_layer_idx=attn_layer_idx)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def spectral_angle_mapper(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = (x * y).sum(dim=1)
    den = (x.norm(dim=1) * y.norm(dim=1)).clamp_min(eps)
    cos = (num / den).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(cos)


def mmd_rbf_tensor(x: torch.Tensor, y: torch.Tensor, sigma2: Optional[torch.Tensor] = None) -> torch.Tensor:
    z = torch.cat([x, y], dim=0)
    if sigma2 is None:
        with torch.no_grad():
            dists = torch.cdist(z, z, p=2) ** 2
            mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
            median_val = dists[mask].median()
            sigma2 = (median_val / 2.0).clamp_min(1e-6)

    K_xx = torch.exp(-torch.cdist(x, x, p=2) ** 2 / (2 * sigma2))
    K_yy = torch.exp(-torch.cdist(y, y, p=2) ** 2 / (2 * sigma2))
    K_xy = torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma2))
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def compute_fixed_sigma2_from_real(real: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        dists = torch.cdist(real, real, p=2) ** 2
        mask = ~torch.eye(real.size(0), dtype=torch.bool, device=real.device)
        median_val = dists[mask].median()
        sigma2 = (median_val / 2.0).clamp_min(1e-6)
    return sigma2


def compute_mmd_rbf_eval_fixedsigma(real: torch.Tensor, fake: torch.Tensor, sigma2: torch.Tensor) -> float:
    return float(mmd_rbf_tensor(real, fake, sigma2=sigma2).detach().item())


def compute_svd_distance(x: torch.Tensor, y: torch.Tensor, top_k: int = 20) -> float:
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)

    cov_x = x_c.t().mm(x_c) / (x_c.size(0) - 1)
    cov_y = y_c.t().mm(y_c) / (y_c.size(0) - 1)

    s_x = torch.linalg.svdvals(cov_x)
    s_y = torch.linalg.svdvals(cov_y)

    k = min(top_k, s_x.size(0), s_y.size(0))
    dist = torch.norm(s_x[:k] - s_y[:k], p=2)
    return float(dist.item())


def compute_prd_precision_recall(real: torch.Tensor, fake: torch.Tensor, quantile: float = 0.1):
    dist_rr = torch.cdist(real, real, p=2)
    mask = ~torch.eye(real.size(0), dtype=torch.bool, device=real.device)
    rr_vals = dist_rr[mask]
    tau = float(rr_vals.quantile(quantile).item())

    dist_rf = torch.cdist(real, fake, p=2)

    min_fake_to_real = dist_rf.min(dim=0).values
    min_real_to_fake = dist_rf.min(dim=1).values

    precision = float((min_fake_to_real <= tau).float().mean().item())
    recall = float((min_real_to_fake <= tau).float().mean().item())
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, tau


def train_wae_transformer(
    excel_path: str,
    device: Optional[str] = None,
    latent_dim: int = 64,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 1e-4,
    lambda_mmd: float = 10.0,
    recon_w_starch: float = 1.0,
    out_dir: str = "wae_transformer_output",
    timestamp_subdir: bool = True,
    eval_n_max: int = 256,
    seed_for_eval: int = 1234,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = make_timestamp_dir(out_dir) if timestamp_subdir else out_dir
    os.makedirs(out_dir, exist_ok=True)

    X_spec, y_starch, spec_min, spec_max, starch_min, starch_max = load_spectra_from_excel(
        excel_path, device=device
    )
    seq_len_spec = X_spec.shape[1]
    seq_len_total = seq_len_spec + 1

    y_norm = (y_starch - starch_min) / (starch_max - starch_min + 1e-8) * 2.0 - 1.0
    XY_norm = torch.cat([X_spec, y_norm], dim=1)

    dataset = JointDataset(XY_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TransformerWAE(
        seq_len_total=seq_len_total,
        latent_dim=latent_dim,
        d_model=96,
        nhead=4,
        enc_layers=2,
        dec_layers=3
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))

    recon_hist, mmdz_hist, total_hist = [], [], []
    mse_hist, sam_hist, mmd_spec_hist, svd_hist = [], [], [], []
    pr_prec_hist, pr_rec_hist, pr_f1_hist = [], [], []

    w = torch.ones(seq_len_total, device=device)
    w[-1] = float(recon_w_starch)

    g = torch.Generator(device=device)
    g.manual_seed(seed_for_eval)

    eval_n = min(eval_n_max, X_spec.shape[0])
    idx_real_eval = torch.randperm(X_spec.shape[0], generator=g, device=device)[:eval_n]
    real_spec_eval_fixed = X_spec[idx_real_eval].detach()
    z_eval_fixed = torch.randn(eval_n, latent_dim, generator=g, device=device)

    sigma2_spec_fixed = compute_fixed_sigma2_from_real(real_spec_eval_fixed)

    for epoch in range(1, epochs + 1):
        model.train()
        ep_recon, ep_mmdz, ep_total = [], [], []

        for real_xy in dataloader:
            real_xy = real_xy.to(device)

            z = model.encode(real_xy)
            recon_xy = model.decode(z)

            recon_loss = ((recon_xy - real_xy) ** 2 * w.unsqueeze(0)).mean()

            z_prior = torch.randn_like(z)
            mmd_z = mmd_rbf_tensor(z, z_prior)

            loss = recon_loss + lambda_mmd * mmd_z

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_recon.append(float(recon_loss.item()))
            ep_mmdz.append(float(mmd_z.item()))
            ep_total.append(float(loss.item()))

        recon_epoch = float(np.mean(ep_recon))
        mmdz_epoch = float(np.mean(ep_mmdz))
        total_epoch = float(np.mean(ep_total))
        recon_hist.append(recon_epoch)
        mmdz_hist.append(mmdz_epoch)
        total_hist.append(total_epoch)

        model.eval()
        with torch.no_grad():
            gen_xy_eval = model.decode(z_eval_fixed)
            gen_spec_eval = gen_xy_eval[:, :seq_len_spec]
            real_spec_eval = real_spec_eval_fixed

            real_mean = real_spec_eval.mean(dim=0, keepdim=True)
            gen_mean = gen_spec_eval.mean(dim=0, keepdim=True)

            mse_val = F.mse_loss(gen_mean, real_mean).item()
            sam_val = spectral_angle_mapper(gen_mean, real_mean)[0].item()

            mmd_spec = compute_mmd_rbf_eval_fixedsigma(real_spec_eval, gen_spec_eval, sigma2_spec_fixed)
            svd_val = compute_svd_distance(real_spec_eval, gen_spec_eval, top_k=20)

            pr_prec, pr_rec, pr_f1, tau = compute_prd_precision_recall(
                real_spec_eval, gen_spec_eval, quantile=0.1
            )

        mse_hist.append(mse_val)
        sam_hist.append(sam_val)
        mmd_spec_hist.append(mmd_spec)
        svd_hist.append(svd_val)
        pr_prec_hist.append(pr_prec)
        pr_rec_hist.append(pr_rec)
        pr_f1_hist.append(pr_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"Recon: {recon_epoch:.4e} | MMD(z): {mmdz_epoch:.4e} | Total: {total_epoch:.4e} | "
            f"MSE(mean): {mse_val:.3e} | SAM(mean): {sam_val:.3e} | "
            f"MMD(spec): {mmd_spec:.3e} | SVD: {svd_val:.3e} | "
            f"PR(prec/rec/F1): {pr_prec:.3f}/{pr_rec:.3f}/{pr_f1:.3f}"
        )

    epochs_arr = np.arange(1, epochs + 1, dtype=np.int32)
    metrics = np.column_stack([
        epochs_arr,
        recon_hist,
        mmdz_hist,
        total_hist,
        mse_hist,
        sam_hist,
        mmd_spec_hist,
        svd_hist,
        pr_prec_hist,
        pr_rec_hist,
        pr_f1_hist
    ])
    metrics_path = os.path.join(out_dir, "training_metrics.csv")
    header = "epoch,recon_loss,mmd_z,total_loss,MSE_mean,SAM_mean,MMD_spec,SVD_dist,PR_precision,PR_recall,PR_F1"
    np.savetxt(metrics_path, metrics, delimiter=",", header=header, comments="")
    print(f"Saved training metrics to: {metrics_path}")

    def plot_metric(values, ylabel, title, fname):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_arr, values, marker="o", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved figure: {path}")

    plot_metric(recon_hist, "Recon loss (weighted MSE)", "Reconstruction loss over epochs", "recon_over_epochs.png")
    plot_metric(mmdz_hist, "MMD(z)", "Latent MMD over epochs", "mmdz_over_epochs.png")
    plot_metric(total_hist, "Total loss", "Total loss over epochs", "total_over_epochs.png")

    plot_metric(mse_hist, "MSE between mean real & generated spectra", "MSE(mean) over epochs", "mse_over_epochs.png")
    plot_metric(sam_hist, "SAM (radians) between mean real & generated", "SAM(mean) over epochs", "sam_over_epochs.png")
    plot_metric(mmd_spec_hist, "MMD (RBF kernel) on spectra", "MMD(spec) over epochs", "mmd_spec_over_epochs.png")
    plot_metric(svd_hist, "SVD-based distance", "SVD distance over epochs", "svd_over_epochs.png")
    plot_metric(pr_f1_hist, "PR F1-score", "PRD F1 over epochs", "prd_f1_over_epochs.png")

    return model, out_dir, (X_spec, y_starch, spec_min, spec_max, starch_min, starch_max, seq_len_spec, seq_len_total)


def export_attention_and_heatmaps(
    model: TransformerWAE,
    latent_dim: int,
    module: str = "decoder",
    layer_idx: int = 0,
    device: str = "cpu",
    out_dir: str = "attn_export"
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    module = module.lower().strip()
    if module not in ["encoder", "decoder"]:
        raise ValueError("module must be 'encoder' or 'decoder'")

    with torch.no_grad():
        if module == "decoder":
            z = torch.randn(1, latent_dim, device=device)
            _ = model.decode(z, attn_layer_idx=layer_idx)
            blk = model.decoder.blocks[layer_idx]
        else:
            z = torch.randn(1, latent_dim, device=device)
            x = model.decode(z)
            _ = model.encode(x, attn_layer_idx=layer_idx)
            blk = model.encoder.blocks[layer_idx]

    attn = blk.last_attn
    if attn is None:
        print("No attention captured; check layer_idx.")
        return

    attn = attn[0].cpu().numpy()
    heads, L, _ = attn.shape

    excel_path = os.path.join(out_dir, f"{module}_layer{layer_idx}_attn.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for h in range(heads):
            df = pd.DataFrame(
                attn[h],
                index=[f"q_{i}" for i in range(L)],
                columns=[f"k_{j}" for j in range(L)]
            )
            df.to_excel(writer, sheet_name=f"head{h}", index=True)
    print(f"Saved attention matrices to: {excel_path}")

    for h in range(heads):
        plt.figure(figsize=(5, 4))
        plt.imshow(attn[h], aspect="auto", origin="lower")
        plt.colorbar(label="Attention weight")
        plt.xlabel("Key index (bands + starch)")
        plt.ylabel("Query index (bands + starch)")
        plt.title(f"{module.capitalize()} attention | layer {layer_idx}, head {h}")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{module}_layer{layer_idx}_head{h}.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Saved attention heatmap: {png_path}")

    attn_mean = attn.mean(axis=0)
    plt.figure(figsize=(5, 4))
    plt.imshow(attn_mean, aspect="auto", origin="lower")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key index (bands + starch)")
    plt.ylabel("Query index (bands + starch)")
    plt.title(f"{module.capitalize()} attention (mean over {heads} heads) | layer {layer_idx}")
    plt.tight_layout()
    mean_png = os.path.join(out_dir, f"{module}_layer{layer_idx}_mean_heads.png")
    plt.savefig(mean_png, dpi=300)
    plt.close()
    print(f"Saved mean-attention heatmap: {mean_png}")


def export_generated_spectra_and_starch(
    model: TransformerWAE,
    latent_dim: int,
    n_samples: int,
    seq_len_spec: int,
    spec_min: float,
    spec_max: float,
    starch_min: float,
    starch_max: float,
    device: str = "cpu",
    out_path: str = "generated_pairs.xlsx"
):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        gen_xy_norm = model.decode(z)

    gen_xy_norm_np = gen_xy_norm.cpu().numpy()
    gen_spec_norm_np = gen_xy_norm_np[:, :seq_len_spec]
    gen_starch_norm_np = gen_xy_norm_np[:, -1:]

    spec_01 = (gen_spec_norm_np + 1.0) / 2.0
    spec_denorm = spec_01 * (spec_max - spec_min) + spec_min

    starch_01 = (gen_starch_norm_np + 1.0) / 2.0
    starch_denorm = starch_01 * (starch_max - starch_min) + starch_min

    band_cols = [f"band_{i+1}" for i in range(seq_len_spec)]
    df_den = pd.DataFrame(np.concatenate([spec_denorm, starch_denorm], axis=1), columns=band_cols + ["starch"])
    df_norm = pd.DataFrame(np.concatenate([gen_spec_norm_np, gen_starch_norm_np], axis=1),
                           columns=band_cols + ["starch_norm"])

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        df_den.to_excel(writer, sheet_name="denorm_pairs", index=False)
        df_norm.to_excel(writer, sheet_name="norm_-1_1_pairs", index=False)

    print(f"Saved generated spectra + starch to: {out_path}")


if __name__ == "__main__":
    excel_path = r"F:\Desktop\Vis-NIR Spectral + Starch.xlsx"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 64

    model, out_dir, (X_data, y_data, spec_min, spec_max, starch_min, starch_max, seq_len_spec, seq_len_total) = \
        train_wae_transformer(
            excel_path=excel_path,
            device=device,
            latent_dim=latent_dim,
            batch_size=16,
            epochs=200,
            lr=1e-4,
            lambda_mmd=10.0,
            recon_w_starch=1.0,
            out_dir="wae_transformer_output",
            timestamp_subdir=True,
            eval_n_max=256,
            seed_for_eval=1234
        )
    attn_dir = os.path.join(out_dir, "attn_decoder")
    n_dec_layers = len(model.decoder.blocks)

    for layer_idx in range(n_dec_layers):
        export_attention_and_heatmaps(
            model,
            latent_dim=latent_dim,
            module="decoder",
            layer_idx=layer_idx,
            device=device,
            out_dir=attn_dir
        )

    n_gen = X_data.shape[0]
    export_generated_spectra_and_starch(
        model,
        latent_dim=latent_dim,
        n_samples=n_gen,
        seq_len_spec=seq_len_spec,
        spec_min=spec_min,
        spec_max=spec_max,
        starch_min=starch_min,
        starch_max=starch_max,
        device=device,
        out_path=os.path.join(out_dir, "generated_pairs.xlsx")
    )