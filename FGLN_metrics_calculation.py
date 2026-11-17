import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import tifffile
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim_sk

# Import relativi al modello FGLN e ai tool
from architecture import model_generator
from Spectral_demosaicing import ergas_matlab

# ==============================================================================
# 1. FUNZIONI DI METRICA (Aggiornate per 8-bit / 16-bit)
# ==============================================================================
def sam_metric(gt_cube: np.ndarray, pred: np.ndarray) -> float:
    """Calcola il SAM medio in gradi su tensori (H, W, C)."""
    dot_product = np.sum(gt_cube * pred, axis=2)
    gt_norm = np.linalg.norm(gt_cube, axis=2)
    pred_norm = np.linalg.norm(pred, axis=2)
    denom = gt_norm * pred_norm
    
    valid = denom > 0
    cosine = np.zeros_like(denom)
    cosine[valid] = dot_product[valid] / denom[valid]
    
    angles = np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))
    return float(np.mean(angles)) if angles.size else 0.0

def compute_metrics_cropped(gt_crop: np.ndarray, pred_crop: np.ndarray, is_16bit: bool = False) -> dict:
    """
    Calcola le metriche riportando i dati alla scala appropriata.
    Se is_16bit=False: scala [0, 255]
    Se is_16bit=True: scala [0, 65535]
    """
    if is_16bit:
        max_val = 65535.0
        dtype = np.uint16
    else:
        max_val = 255.0
        dtype = np.uint8

    true_img = np.clip(np.round(gt_crop * max_val), 0, max_val).astype(dtype)
    pred_img = np.clip(np.round(pred_crop * max_val), 0, max_val).astype(dtype)
    
    C = true_img.shape[2]
    win_size = min(7, true_img.shape[0], true_img.shape[1])
    if win_size % 2 == 0:
        win_size -= 1

    psnr_vals = [
        peak_signal_noise_ratio(true_img[:, :, b], pred_img[:, :, b], data_range=max_val)
        for b in range(C)
    ]
    ssim_vals = Parallel(n_jobs=-1)(
        delayed(ssim_sk)(pred_img[:, :, b], true_img[:, :, b], data_range=max_val, win_size=win_size)
        for b in range(C)
    )

    return {
        "PSNR":  float(np.mean(psnr_vals)),
        "SSIM":  float(np.mean(ssim_vals)),
        "SAM":   sam_metric(true_img.astype(np.float64), pred_img.astype(np.float64)),
        "ERGAS": ergas_matlab(true_img, pred_img, 1/3),
    }

# ==============================================================================
# 2. LETTURA DATI PER IL MODELLO (Pattern Spaziale Reale Identico al Dataloader)
# ==============================================================================
def load_sample_from_folder(sample_dir: str, msfa_pattern: np.ndarray) -> tuple:
    orig_dir = os.path.join(sample_dir, 'original_bands')
    sparse_dir = os.path.join(sample_dir, 'sparse_matrices')
    
    # Appiattimento del pattern: l'ordine di impilamento sarà [0, 5, 2, 3, 8, 7, 6, 1, 4]
    flat_pattern = msfa_pattern.flatten()

    target_list, sparse_list = [], []
    for band_idx in flat_pattern:
        orig_file = os.path.join(orig_dir, f'original_band_{band_idx}.tiff')
        sparse_file = os.path.join(sparse_dir, f'sparse_matrix_{band_idx}.tiff')
        
        target_img = tifffile.imread(orig_file).astype(np.float32) / 255.0
        sparse_img = tifffile.imread(sparse_file).astype(np.float32) / 255.0
        
        target_list.append(target_img)
        sparse_list.append(sparse_img)

    target = np.stack(target_list, axis=-1)
    sparse_cube = np.stack(sparse_list, axis=-1)

    mosaic_files = glob.glob(os.path.join(sample_dir, '*Mosaic*.tif*'))
    raw = tifffile.imread(mosaic_files[0]).astype(np.float32) / 255.0

    return target, sparse_cube, raw

# ==============================================================================
# 3. FUNZIONE DI FORWARD CON TILING SPECIFICA PER FGLN
# ==============================================================================
def forward_tiled_fgln(model, sparse_tensor: torch.Tensor, raw_tensor: torch.Tensor, msfa_size: int, tile_size: int, overlap: int) -> np.ndarray:
    """
    Esegue un forward pass basato sul tiling per superare limiti VRAM su immagini ad alta risoluzione.
    FGLN richiede: input_raw [B, 1, H, W] e input_sparse_raw [B, C, H, W].
    """
    device = sparse_tensor.device
    N, C, H, W = sparse_tensor.shape
    tile_size = tile_size - (tile_size % msfa_size)
    overlap = overlap - (overlap % msfa_size)
    stride = tile_size - overlap
    half_overlap = overlap // 2
    out_full = torch.zeros((N, C, H, W), dtype=torch.float32, device=device)

    for h in range(0, H, stride):
        for w in range(0, W, stride):
            h_end, w_end = min(h + tile_size, H), min(w + tile_size, W)
            th, tw = h_end - h, w_end - w
            
            sparse_tile = sparse_tensor[:, :, h:h_end, w:w_end]
            raw_tile = raw_tensor[:, :, h:h_end, w:w_end]

            with torch.no_grad():
                # L'ordine di inferenza standard del file di train: model(input_raw, input_sparse_raw)
                full_tile = model(raw_tile, sparse_tile)

            crop_top = 0 if h == 0 else half_overlap
            crop_bottom = th if h_end == H else th - half_overlap
            crop_left = 0 if w == 0 else half_overlap
            crop_right = tw if w_end == W else tw - half_overlap

            out_full[:, :, h + crop_top:h + crop_bottom, w + crop_left:w + crop_right] = \
                full_tile[:, :, crop_top:crop_bottom, crop_left:crop_right].cpu()

    return out_full.squeeze(0).cpu().numpy().transpose(1, 2, 0)

# ==============================================================================
# 4. PIPELINE PRINCIPALE DI VALUTAZIONE
# ================================================S==============================
def main():
    SIMULATE_16BIT = False # <-- TOGGLE 8-bit / 16-bit
    
    msfa_size = 3
    epoch = 3570
    test_dir = "/mnt/Volume_Interno/UBUNTU/dataset_flavia/valid" # path definito nell'args test_dir
    checkpoint_fgln = f"train_exp/My/2026_09_29_21_27_45/net_{epoch}epoch.pth" # TODO: Aggiorna con il tuo path esatto
    
    # Adatta il nome del file di output in base al bit-depth
    bit_suffix = "16bit" if SIMULATE_16BIT else "8bit"
    output_csv = f"results/benchmark/FGLN_Evaluation_ExactPixels_{epoch}_valid_{bit_suffix}.csv"

    # Pattern rigoroso allineato alla disposizione spaziale fisica (GenericFolderDataset)
    pattern = np.array([
        [0, 5, 2],
        [3, 8, 7],
        [6, 1, 4]
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Caricamento Modello FGLN (Valutazione {bit_suffix})...")

    # Istanza del modello come definito nell'argparse
    model = model_generator('FGLN', msfa_size, pretrained_model_path=None).to(device)
    
    # Caricamento Checkpoint
    checkpoint = torch.load(checkpoint_fgln, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    sample_dirs = sorted([d.path for d in os.scandir(test_dir) if d.is_dir()])
    results = []

    for sample_dir in tqdm(sample_dirs, desc=f"Valutazione FGLN ({bit_suffix})"):
        sample_name = os.path.basename(sample_dir)
        
        # Caricamento e normalizzazione in [0, 1]
        img_np, sparse_np_full, raw_np_full = load_sample_from_folder(sample_dir, pattern)
        H_orig, W_orig, _ = img_np.shape

        # Mantenimento margini spaziali multipli di 3 per un tiling spaziale corretto
        m0 = int(np.fix(H_orig / 3) * 3)
        m1 = int(np.fix(W_orig / 3) * 3)
        
        sparse_np_net = sparse_np_full[:m0, :m1, :]
        raw_np_net = raw_np_full[:m0, :m1]
        
        # Trasformazione in tensori PyTorch
        sparse_tensor = torch.from_numpy(np.ascontiguousarray(sparse_np_net.transpose(2, 0, 1))).unsqueeze(0).to(device)
        raw_tensor = torch.from_numpy(np.ascontiguousarray(raw_np_net)).unsqueeze(0).unsqueeze(0).to(device)

        # Inferenza Tiled per FGLN
        fgln_pred = forward_tiled_fgln(model, sparse_tensor, raw_tensor, msfa_size=msfa_size, tile_size=1200, overlap=60)
        fgln_pred = np.clip(fgln_pred, 0.0, 1.0)

        # Ritaglio dinamico speculare alle metriche per le Baseline (-2)
        gt_crop = img_np[2 : m0 - 2, 2 : m1 - 2]
        fgln_crop = fgln_pred[2 : m0 - 2, 2 : m1 - 2]

        # Calcolo metriche gestito dal toggle
        metrics = compute_metrics_cropped(gt_crop, fgln_crop, is_16bit=SIMULATE_16BIT)

        eval_H, eval_W = gt_crop.shape[:2]
        results.append({
            "Filename": sample_name,
            "Resolution_Eval": f"{eval_H}x{eval_W}",
            "FGLN_PSNR": round(metrics["PSNR"], 4),
            "FGLN_SSIM": round(metrics["SSIM"], 4),
            "FGLN_SAM": round(metrics["SAM"], 4),
            "FGLN_ERGAS": round(metrics["ERGAS"], 4)
        })

    # Aggregazione e Salvataggio
    if results:
        df = pd.DataFrame(results)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        mean_row = {col: round(df[col].mean(), 4) for col in numeric_cols}
        mean_row["Filename"] = "MEDIA GLOBALE"
        mean_row["Resolution_Eval"] = "-"

        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSalvataggio metriche FGLN completato in: {output_csv}")

if __name__ == "__main__":
    main()