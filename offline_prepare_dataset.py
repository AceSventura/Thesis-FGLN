import os
import tifffile
import numpy as np
from tqdm import tqdm

def prepare_mmap_dataset(image_dir, msfa_pattern):
    sample_dirs = [os.path.join(image_dir, d) for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    for sample_dir in tqdm(sample_dirs, desc="Generazione File Memmap"):
        orig_dir = os.path.join(sample_dir, 'original_bands')
        sparse_dir = os.path.join(sample_dir, 'sparse_matrices')
        
        target_list, sparse_list = [], []
        
        for band_idx in msfa_pattern:
            # tifffile legge l'immagine mantenendo il dtype originale (es. uint8)
            target_img = tifffile.imread(os.path.join(orig_dir, f'original_band_{band_idx}.tiff'))
            sparse_img = tifffile.imread(os.path.join(sparse_dir, f'sparse_matrix_{band_idx}.tiff'))
            
            target_list.append(target_img)
            sparse_list.append(sparse_img)

        # Stacking
        target_full = np.stack(target_list, axis=-1)
        sparse_full = np.stack(sparse_list, axis=-1)
        
        # Raw / Mosaic
        mosaic_file = [f for f in os.listdir(sample_dir) if 'Mosaic' in f and f.endswith(('.tif', '.tiff'))][0]
        raw_full = tifffile.imread(os.path.join(sample_dir, mosaic_file))

        # Salvataggio su disco (non usare np.savez, serve .npy puro per il mmap)
        np.save(os.path.join(sample_dir, 'target_stacked.npy'), target_full)
        np.save(os.path.join(sample_dir, 'input_stacked.npy'), sparse_full)
        np.save(os.path.join(sample_dir, 'raw_stacked.npy'), raw_full)

if __name__ == "__main__":
    pattern = np.array([
        [0, 5, 2],
        [3, 8, 7],
        [6, 1, 4]
    ]).flatten()
    
    print("Processando TRAIN...")
    prepare_mmap_dataset("/mnt/Volume_Interno/UBUNTU/dataset_flavia/train", pattern)
    print("Processando VALID...")
    prepare_mmap_dataset("/mnt/Volume_Interno/UBUNTU/dataset_flavia/valid", pattern)