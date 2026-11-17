import os
import glob
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile
from tqdm import tqdm

from architecture import model_generator                  # FGLN
from student_model_attention import StudentDemosaic       # mini-FGLN (Student)

# ==============================================================================
# 0. CONFIGURAZIONE
# ==============================================================================
MSFA_SIZE = 3
NUM_BANDS = 9

PATTERN = np.array([
    [0, 5, 2],
    [3, 8, 7],
    [6, 1, 4],
])


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/mnt/Volume_Interno/UBUNTU/dataset_flavia/valid")
    p.add_argument("--fgln_ckpt", default="train_exp/My/2026_09_29_21_27_45/net_3570epoch.pth")
    p.add_argument("--student_ckpt",
                   default="checkpoint_student/Flavia_StudentOfflineKD_FGLN_3_32_261002_191244/Student_model_latest.pth")
    p.add_argument("--student_channels", type=int, default=32)
    p.add_argument("--student_blocks", type=int, default=3)
    p.add_argument("--tile_size", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--warmup_runs", type=int, default=3, help="Esecuzioni di hot start PER OGNI immagine e metodo")
    p.add_argument("--n_runs", type=int, default=20, help="Esecuzioni misurate PER OGNI immagine e metodo")
    p.add_argument("--max_images", type=int, default=None, help="Limita il numero di immagini (debug)")
    p.add_argument("--out_dir", default="results/benchmark")
    return p.parse_args()


# ==============================================================================
# 1. CARICAMENTO DATI (una immagine alla volta, FUORI dal tempo misurato)
# ==============================================================================
def load_sample(sample_dir: str, msfa_pattern: np.ndarray, device):
    """raw [1,1,H,W] e sparse [1,C,H,W] già sulla GPU, float32 in [0,1],
    ritagliati a multipli di 3."""
    sparse_dir = os.path.join(sample_dir, "sparse_matrices")
    sparse_list = []
    for band_idx in msfa_pattern.flatten():
        f = os.path.join(sparse_dir, f"sparse_matrix_{band_idx}.tiff")
        sparse_list.append(tifffile.imread(f).astype(np.float32) / 255.0)
    sparse = np.stack(sparse_list, axis=0)  # C,H,W

    mosaic_files = glob.glob(os.path.join(sample_dir, "*Mosaic*.tif*"))
    raw = tifffile.imread(mosaic_files[0]).astype(np.float32) / 255.0  # H,W

    H, W = raw.shape
    m0 = (H // MSFA_SIZE) * MSFA_SIZE
    m1 = (W // MSFA_SIZE) * MSFA_SIZE
    sparse = np.ascontiguousarray(sparse[:, :m0, :m1])
    raw = np.ascontiguousarray(raw[:m0, :m1])

    return {
        "raw": torch.from_numpy(raw)[None, None].to(device),
        "sparse": torch.from_numpy(sparse)[None].to(device),
    }


# ==============================================================================
# 2. METODI BASELINE SU GPU: WB e ID
# ==============================================================================
def build_kernel_5x5(device):
    v = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    k = (1 / 3) * np.outer(v, v) * (1 / 3)
    return torch.from_numpy(k.astype(np.float32)).to(device)


def build_masks_gpu(H, W, device):
    """Maschere [1,9,H,W] equivalenti a generate_mask(): banda k -> mask_idx."""
    band_to_mask = [0, 5, 2, 3, 8, 7, 6, 1, 4]
    masks = torch.zeros((1, NUM_BANDS, H, W), dtype=torch.float32, device=device)
    for k in range(NUM_BANDS):
        m = band_to_mask[k]
        masks[0, k, m // 3::3, m % 3::3] = 1.0
    return masks


@torch.inference_mode()
def wb_forward_gpu(sparse: torch.Tensor, kernel5: torch.Tensor) -> torch.Tensor:
    """WB (convolve2d mode='same', zero-padding) su GPU: conv depthwise 5x5.
    sparse: [1,9,H,W] -> [1,9,H,W]. Il kernel è simmetrico, quindi
    correlazione e convoluzione coincidono."""
    k5 = kernel5.view(1, 1, 5, 5).repeat(NUM_BANDS, 1, 1, 1)
    return F.conv2d(sparse, k5, padding=2, groups=NUM_BANDS)


@torch.inference_mode()
def id_forward_gpu(raw: torch.Tensor, kernel5: torch.Tensor) -> torch.Tensor:
    """ID su GPU. raw: [1,1,H,W] -> [1,9,H,W].
    Il padding 'reflect' di torch differisce da quello di scipy solo sui bordi
    (< 3 px), che vengono comunque ritagliati nelle metriche."""
    device = raw.device
    _, _, H, W = raw.shape
    masks = build_masks_gpu(H, W, device)

    k3 = torch.full((1, 1, 3, 3), 1.0 / 9.0, device=device)
    I_M = F.conv2d(F.pad(raw, (1, 1, 1, 1), mode="reflect"), k3)

    delta = (raw - I_M) * masks
    k5 = kernel5.view(1, 1, 5, 5).repeat(NUM_BANDS, 1, 1, 1)
    delta_stim = F.conv2d(F.pad(delta, (2, 2, 2, 2), mode="reflect"), k5, groups=NUM_BANDS)
    return I_M + delta_stim


# ==============================================================================
# 3. TILING GENERICO (stessa logica per FGLN e mini-FGLN)
# ==============================================================================
@torch.inference_mode()
def forward_tiled(model_fn, raw, sparse, tile_size, overlap):
    """model_fn(raw_tile, sparse_tile) -> [N,C,h,w]. Tensori già su GPU."""
    N, C, H, W = sparse.shape
    tile_size = tile_size - (tile_size % MSFA_SIZE)
    overlap = overlap - (overlap % MSFA_SIZE)
    stride = tile_size - overlap
    half = overlap // 2
    out = torch.zeros((N, C, H, W), dtype=torch.float32, device=sparse.device)

    for h in range(0, H, stride):
        for w in range(0, W, stride):
            h_end, w_end = min(h + tile_size, H), min(w + tile_size, W)
            th, tw = h_end - h, w_end - w
            full_tile = model_fn(raw[:, :, h:h_end, w:w_end], sparse[:, :, h:h_end, w:w_end])

            ct = 0 if h == 0 else half
            cb = th if h_end == H else th - half
            cl = 0 if w == 0 else half
            cr = tw if w_end == W else tw - half
            out[:, :, h + ct:h + cb, w + cl:w + cr] = full_tile[:, :, ct:cb, cl:cr]
    return out


# ==============================================================================
# 4. PIPELINE (solo calcolo: input su GPU, output lasciato su GPU)
# ==============================================================================
def make_pipeline(name, model=None, kernel5=None, tile_size=1200, overlap=60):
    if name == "ID":
        def run(s):
            return id_forward_gpu(s["raw"], kernel5).clamp(0, 1)
    elif name == "WB":
        def run(s):
            return wb_forward_gpu(s["sparse"], kernel5).clamp(0, 1)
    elif name == "FGLN":
        def run(s):
            return forward_tiled(lambda r, sp: model(r, sp), s["raw"], s["sparse"],
                                 tile_size, overlap).clamp(0, 1)
    elif name == "mini-FGLN":
        def run(s):
            return forward_tiled(lambda r, sp: model(sp), s["raw"], s["sparse"],
                                 tile_size, overlap).clamp(0, 1)
    else:
        raise ValueError(name)
    return run


# ==============================================================================
# 5. MISURE SU SINGOLA IMMAGINE
# ==============================================================================
def sync(device):
    torch.cuda.synchronize(device)


def time_runs(run_fn, sample, n_runs, device):
    """Tempi (ms) di n_runs esecuzioni consecutive sulla stessa immagine."""
    ts = []
    for _ in range(n_runs):
        sync(device)
        t0 = time.perf_counter()
        out = run_fn(sample)
        sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
        del out
    return np.array(ts)


def peak_extra_mb(run_fn, sample, device):
    """Memoria GPU AGGIUNTIVA usata dal metodo su questa immagine:
    picco - memoria già allocata prima della chiamata (pesi dei modelli e
    immagine di input esclusi). L'output è incluso."""
    torch.cuda.empty_cache()
    sync(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = run_fn(sample)
    sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    del out
    return (peak - base) / 1024 ** 2


def summarize(df_img, params, ref_method="FGLN"):
    g = df_img.groupby("Method", sort=False)
    s = g.agg(
        Time_ms_mean=("Mean_ms", "mean"),
        Time_ms_std_across_images=("Mean_ms", "std"),
        RunToRun_std_ms=("Std_ms", "mean"),
        Median_ms=("Median_ms", "mean"),
        Min_ms=("Min_ms", "mean"),
        Peak_extra_MB_mean=("Peak_extra_MB", "mean"),
        Peak_extra_MB_max=("Peak_extra_MB", "max"),
        Mpx=("Mpx", "mean"),
        N_images=("Filename", "count"),
    ).reset_index()
    s["Throughput_Mpx_per_s"] = s["Mpx"] / (s["Time_ms_mean"] / 1000.0)
    s.insert(1, "Params", s["Method"].map(params))
    if ref_method in set(s["Method"]):
        ref = s.loc[s.Method == ref_method, "Time_ms_mean"].iloc[0]
        s["Speedup_vs_FGLN"] = ref / s["Time_ms_mean"]
    return s.drop(columns=["Mpx"]).round(4)


# ==============================================================================
# 6. MAIN
# ==============================================================================
def load_fgln(path, device):
    model = model_generator("FGLN", MSFA_SIZE, pretrained_model_path=None).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model.eval()


def load_student(path, channels, blocks, device):
    model = StudentDemosaic(msfa_size=MSFA_SIZE, channels=channels, num_blocks=blocks).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.eval()


def n_params(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA non disponibile: il benchmark richiede una GPU.")
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    sample_dirs = sorted([d.path for d in os.scandir(args.data_dir) if d.is_dir()])
    if args.max_images:
        sample_dirs = sample_dirs[:args.max_images]

    fgln = load_fgln(args.fgln_ckpt, device)
    student = load_student(args.student_ckpt, args.student_channels, args.student_blocks, device)
    kernel5 = build_kernel_5x5(device)

    specs = [("WB", None), ("ID", None), ("FGLN", fgln), ("mini-FGLN", student)]
    params = {name: n_params(m) for name, m in specs}
    pipelines = {
        name: make_pipeline(name, model=m, kernel5=kernel5,
                            tile_size=args.tile_size, overlap=args.overlap)
        for name, m in specs
    }

    rows = []
    for sample_dir in tqdm(sample_dirs, desc="Benchmark per immagine"):
        name = os.path.basename(sample_dir)
        sample = load_sample(sample_dir, PATTERN, device)
        H, W = sample["raw"].shape[-2:]

        for method, _ in specs:
            run_fn = pipelines[method]

            # Hot start sulla singola immagine (stesse forme dei tile della misura)
            for _ in range(args.warmup_runs):
                del_out = run_fn(sample)
                del del_out
            sync(device)

            mem = peak_extra_mb(run_fn, sample, device)
            ts = time_runs(run_fn, sample, args.n_runs, device)

            rows.append({
                "Filename": name,
                "Resolution": f"{H}x{W}",
                "Mpx": H * W / 1e6,
                "Method": method,
                "Mean_ms": ts.mean(),
                "Std_ms": ts.std(ddof=1) if len(ts) > 1 else 0.0,
                "Median_ms": float(np.median(ts)),
                "Min_ms": ts.min(),
                "Max_ms": ts.max(),
                "Peak_extra_MB": mem,
            })

        del sample
        torch.cuda.empty_cache()

    df_img = pd.DataFrame(rows)
    df_sum = summarize(df_img, params)

    os.makedirs(args.out_dir, exist_ok=True)
    df_img.round(4).to_csv(os.path.join(args.out_dir, "timing_per_image_gpu.csv"), index=False)
    df_sum.to_csv(os.path.join(args.out_dir, "timing_summary_gpu.csv"), index=False)

    print("\n" + df_sum.to_string(index=False))
    print(f"\nSalvato in: {args.out_dir}/timing_per_image_gpu.csv e timing_summary_gpu.csv")


if __name__ == "__main__":
    main()