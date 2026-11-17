import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import dei modelli e funzioni (non importiamo più FGLN)
from student_model_attention import StudentDemosaic
from student_encoderDecoder import EncoderDecoderStudent
from GenericFolderDataset import GenericFolderDataset
from Spectral_demosaicing import get_datetime_str

def charbonnier_loss(pred, target, eps=1e-3):
    """
    Loss Spaziale: Charbonnier loss globale tra tensore calcolato e ground truth (Teacher)
    """
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def frequency_loss(pred, target):
    fft_pred = torch.fft.fft2(pred, dim=(-2, -1), norm='ortho')
    fft_target = torch.fft.fft2(target, dim=(-2, -1), norm='ortho')
    return charbonnier_loss(torch.view_as_real(fft_pred), torch.view_as_real(fft_target))

def calculate_sei(pred, msfa_size):
    unshuffled = F.pixel_unshuffle(pred, msfa_size)
    spatial_means = unshuffled.mean(dim=(2, 3))
    spatial_means = spatial_means.view(pred.size(0), pred.size(1), msfa_size**2)
    band_var = spatial_means.var(dim=2, unbiased=False)
    return band_var.mean().item()

def calculate_ssim(img1, img2, val_range=1.0):
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    channels = img1.size(1)
    kernel_size = 11
    sigma = 1.5
    coords = torch.arange(kernel_size, dtype=img1.dtype, device=img1.device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    mu1 = F.conv2d(img1, kernel, padding=kernel_size//2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=kernel_size//2, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2, groups=channels) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean().item()


parser = argparse.ArgumentParser(description="Offline Knowledge Distillation for Spectral Demosaicing")
parser.add_argument("--cuda",               action="store_false", help="Use cuda?")
parser.add_argument("--threads",            type=int,   default=4)
# NOTA: Ora puoi aumentare drasticamente il batchSize (es. 256 o 512) dato che il FGLN non occupa VRAM
parser.add_argument("--batchSize",          type=int,   default=128)
parser.add_argument("--msfa_size",          type=int,   default=3)
parser.add_argument("--dataset",            default="Flavia", type=str)
parser.add_argument("--resume",             default="", type=str)
parser.add_argument("--start-epoch",        default=1,  type=int)
parser.add_argument("--lr",                 default=2e-4, type=float)
parser.add_argument("--channels",           type=int,   default=32)
parser.add_argument("--blocks",             type=int,   default=4)
parser.add_argument("--nEpochs",            default=1000, type=int)
parser.add_argument("--total-epochs",       default=0,  type=int)


def _print_pattern_banner(msfa_pattern):
    print("=" * 60)
    print("[PATTERN] Pattern MSFA attivo:")
    print(msfa_pattern)
    print("=" * 60)


def main():
    global opt
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    opt = parser.parse_args()
    opt.norm_flag = False
    opt.dt = get_datetime_str()
    name = "_StudentOfflineKD_FGLN_"
    #name = "_EncoderDecoderStudent_"
    opt.save_dir = (
        'checkpoint_student/' + opt.dataset
        + name + str(opt.blocks) + '_' + str(opt.channels)
        + "_" + opt.dt + '/'
    )

    if not torch.cuda.is_available() and opt.cuda:
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(2812)
    if opt.cuda:
        torch.cuda.manual_seed(2812)
    cudnn.benchmark = True

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    msfa_pattern = np.array([
        [0, 5, 2],
        [3, 8, 7],
        [6, 1, 4]
    ])
    _print_pattern_banner(msfa_pattern)

    flat_pattern = msfa_pattern.flatten().tolist()

    train_set_dir = "/mnt/Volume_Interno/UBUNTU/dataset_flavia/train"
    test_set_dir  = "/mnt/Volume_Interno/UBUNTU/dataset_flavia/valid"

    train_set = GenericFolderDataset(
        image_dir=train_set_dir,
        msfa_size=opt.msfa_size,
        msfa_pattern=flat_pattern,
        norm_flag=True,
        patch_size=90,
        augment=False,
        type='train',
        use_offline_kd=True  # Abilitato
    )

    test_set = GenericFolderDataset(
        image_dir=test_set_dir,
        msfa_size=opt.msfa_size,
        msfa_pattern=flat_pattern,
        norm_flag=True,
        patch_size=510,
        augment=False,
        type='test',
        use_offline_kd=True  # Abilitato
    )

    training_data_loader = DataLoader(
        dataset=train_set, batch_size=opt.batchSize, shuffle=True,
        num_workers=opt.threads, pin_memory=True)
    testing_data_loader = DataLoader(
        dataset=test_set, batch_size=2, shuffle=False,
        num_workers=2, pin_memory=True)

    # === IL TEACHER NON VIENE PIU' CARICATO QUI ===
    
    model_student = StudentDemosaic(
        msfa_size=opt.msfa_size,
        channels=opt.channels,
        num_blocks=opt.blocks
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model_student.parameters()),
        lr=opt.lr)
    
    scaler = None

    if opt.resume and os.path.isfile(opt.resume):
        checkpoint_student = torch.load(opt.resume, map_location=lambda storage, loc: storage, weights_only=False)
        opt.start_epoch = checkpoint_student["epoch"] + 1

        saved_model = checkpoint_student["model"]
        if isinstance(saved_model, nn.Module):
            model_student.load_state_dict(saved_model.state_dict())
        else:
            model_student.load_state_dict(saved_model)

        if "optimizer" in checkpoint_student:
            optimizer.load_state_dict(checkpoint_student["optimizer"])
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', opt.lr)

        if "scaler" in checkpoint_student and scaler is not None and checkpoint_student["scaler"] is not None:
            scaler.load_state_dict(checkpoint_student["scaler"])

        opt.save_dir = os.path.dirname(opt.resume) + '/'
        print(f"[RESUME] Ripreso da epoch {opt.start_epoch - 1}: {opt.resume}")

    if opt.total_epochs == 0:
        opt.total_epochs = (opt.start_epoch - 1) + opt.nEpochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.nEpochs, eta_min=1e-6, last_epoch=opt.start_epoch - 2,
    )

    if opt.resume and os.path.isfile(opt.resume):
        checkpoint_student = torch.load(opt.resume, map_location=lambda storage, loc: storage, weights_only=False)
        if "scheduler" in checkpoint_student and checkpoint_student["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint_student["scheduler"])

    if opt.cuda:
        model_student = model_student.cuda()

    results = {
        'spatial_loss': [], 'freq_loss': [], 'all_loss': [], 'valid_loss': [],
        'psnr': [], 'sei': [], 'teacher_student_ssim': [],
    }
    best_valid_loss = float('inf')
    best_psnr = float('-inf')

    csv_path = os.path.join(opt.save_dir, 'training_metrics.csv')
    if opt.resume and os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_old = df_old[df_old['Epoch'] < opt.start_epoch]
        storico_epoche = len(df_old)
        for col in results.keys():
            results[col] = df_old[col].tolist() if col in df_old.columns else [float('nan')] * storico_epoche
        if 'valid_loss' in df_old.columns and not df_old.empty:
            best_valid_loss = df_old['valid_loss'].min()
        if 'psnr' in df_old.columns and not df_old.empty:
            best_psnr = df_old['psnr'].max()

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{opt.nEpochs} — LR: {current_lr:.2e}")

        running_results = train_kd(
            training_data_loader, optimizer, scheduler,
            model_student,
            epoch, opt.nEpochs, opt, scaler,
        )
        
        is_best_loss = False
        is_best_psnr = False
        current_valid_loss = float('nan')
        current_psnr = float('nan')
        current_sei = float('nan')
        current_ts_ssim = float('nan')

        # Test ogni 5 epoche invece che 10
        if epoch % 5 == 0:
            test_results = test(
                testing_data_loader,
                model_student,
                epoch, opt.nEpochs, opt.msfa_size, opt,
            )
            current_valid_loss = test_results['valid_loss']
            current_psnr = test_results['psnr']
            current_sei = test_results['sei']
            current_ts_ssim = test_results['teacher_student_ssim']

            if current_valid_loss < best_valid_loss:
                best_valid_loss = current_valid_loss
                is_best_loss = True
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                is_best_psnr = True

        results['spatial_loss'].append(running_results['spatial_loss'])
        results['freq_loss'].append(running_results['freq_loss'])
        results['all_loss'].append(running_results['all_loss'])
        results['valid_loss'].append(current_valid_loss)
        results['psnr'].append(current_psnr)
        results['sei'].append(current_sei)
        results['teacher_student_ssim'].append(current_ts_ssim)

        save_checkpoint(model_student, optimizer, scheduler, scaler,
                        epoch, opt.save_dir, is_best_loss, is_best_psnr)
        save_statistics(results, epoch, opt.start_epoch, opt.save_dir)


def train_kd(training_data_loader, optimizer, scheduler,
             model_student,
             epoch, num_epochs, opt, scaler):
    train_bar = tqdm(training_data_loader)
    running_results = {'batch_sizes': 0, 'spatial_loss': 0, 'freq_loss': 0, 'all_loss': 0}

    model_student.train()

    lambda_spatial = 1.0
    lambda_freq = 0.25  

    for batch in train_bar:
        # Recupero di tutti e 4 i tensori (teacher_out è l'immagine pre-computata dal FGLN)
        input_raw, input_tens, target_reale, teacher_out = batch[0], batch[1], batch[2], batch[3]
        N = input_tens.size(0)
        running_results['batch_sizes'] += N

        if opt.cuda:
            input_tens = input_tens.cuda(non_blocking=True)
            #input_raw = input_raw.cuda(non_blocking=True)
            teacher_out = teacher_out.cuda(non_blocking=True)
            # input_raw e target_reale non servono su GPU durante la distillazione spaziale

        # Convertiamo il teacher (float16) in float32 per stabilità numerica
        teacher_out = teacher_out.float()
        
        # Riapplichiamo il clamp per sicurezza matematica
        teacher_out = torch.clamp(teacher_out * 255.0, min=0.0, max=255.0) / 255.0

        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                student_out = model_student(input_tens)
                
                loss_spatial = charbonnier_loss(student_out, teacher_out)
                loss_freq = frequency_loss(student_out, teacher_out)
                loss = (lambda_spatial * loss_spatial) + (lambda_freq * loss_freq)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            student_out = model_student(input_tens)
            
            loss_spatial = charbonnier_loss(student_out, teacher_out)
            loss_freq = frequency_loss(student_out, teacher_out)
            loss = (lambda_spatial * loss_spatial) + (lambda_freq * loss_freq)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
            optimizer.step()

        running_results['spatial_loss'] += loss_spatial.item() * N
        running_results['freq_loss'] += loss_freq.item() * N
        running_results['all_loss'] += loss.item() * N

        train_bar.set_description(
            '[%d/%d] L_Spa: %.4f L_Freq: %.4f L_Tot: %.4f' % (
                epoch, num_epochs,
                running_results['spatial_loss'] / running_results['batch_sizes'],
                running_results['freq_loss'] / running_results['batch_sizes'],
                running_results['all_loss'] / running_results['batch_sizes']))

    scheduler.step()

    return {
        'spatial_loss':     running_results['spatial_loss'] / running_results['batch_sizes'],
        'freq_loss':        running_results['freq_loss'] / running_results['batch_sizes'],
        'all_loss':         running_results['all_loss'] / running_results['batch_sizes'],
    }


def test(testing_data_loader, model_student,
         epoch, num_epochs, msfa_size, opt):

    test_bar = tqdm(testing_data_loader)
    test_results = {
        'batch_sizes': 0, 'psnr': 0,
        'valid_loss': 0.0, 'sei': 0.0, 'teacher_student_ssim': 0.0,
    }
    model_student.eval()

    with torch.no_grad():
        for batch in test_bar:
            input_raw, input_tens, target, teacher_out = batch[0], batch[1], batch[2], batch[3]

            N, C, H, W = input_tens.size()
            test_results['batch_sizes'] += N

            if opt.cuda:
                input_tens = input_tens.cuda(non_blocking=True)
                #input_raw = input_raw.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                teacher_out = teacher_out.cuda(non_blocking=True)

            teacher_out = teacher_out.float()
            
            HR_4x = model_student(input_tens)

            # Scala e clippa in [0, 255]
            HR_4x = torch.clamp(HR_4x * 255.0, min=0.0, max=255.0)
            teacher_out = torch.clamp(teacher_out * 255.0, min=0.0, max=255.0)
            target = target * 255.0 

            batch_loss = F.mse_loss(HR_4x, target).item()
            test_results['valid_loss'] += batch_loss * N

            batch_sei = calculate_sei(HR_4x / 255.0, msfa_size)
            test_results['sei'] += batch_sei * N

            mse_per_channel = torch.mean((HR_4x - target) ** 2, dim=(2, 3)) 
            psnr_per_channel = 10 * torch.log10((255.0 ** 2) / (mse_per_channel + 1e-8))
            test_results['psnr'] += psnr_per_channel.sum().item()

            batch_ssim = calculate_ssim(HR_4x, teacher_out, val_range=255.0)
            test_results['teacher_student_ssim'] += batch_ssim * N

            test_bar.set_description(
                '[%d/%d] Val Loss: %.6f PSNR: %.2f SEI: %.4e' % (
                    epoch, num_epochs,
                    test_results['valid_loss'] / test_results['batch_sizes'],
                    test_results['psnr'] / test_results['batch_sizes'] / (msfa_size**2),
                    test_results['sei'] / test_results['batch_sizes']))

    return {
        'valid_loss':           test_results['valid_loss'] / test_results['batch_sizes'],
        'psnr':                 test_results['psnr'] / test_results['batch_sizes'] / (msfa_size**2),
        'sei':                  test_results['sei'] / test_results['batch_sizes'],
        'teacher_student_ssim': test_results['teacher_student_ssim'] / test_results['batch_sizes'],
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, save_dir,
                    is_best_loss=False, is_best_psnr=False):
    state = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict() if scaler is not None else None,
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, "Student_model_latest.pth"))
    if epoch % 10 == 0:
        torch.save(state, os.path.join(save_dir, f"Student_model_epoch_{epoch}.pth"))
    if is_best_loss:
        torch.save(state, os.path.join(save_dir, "Student_model_best_loss.pth"))
    if is_best_psnr:
        torch.save(state, os.path.join(save_dir, "Student_model_best_psnr.pth"))


def save_statistics(results, epoch, start_epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    total_epochs = len(results['psnr'])
    df = pd.DataFrame(data=results, index=range(1, total_epochs + 1))
    df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index_label='Epoch')


if __name__ == "__main__":
    main()