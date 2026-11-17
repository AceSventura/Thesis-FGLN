import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from tqdm import tqdm
from utils import *
from architecture import *
import datetime
from focal_frequency_loss import FocalFrequencyLoss

# --- Importa il tuo Dataset ---
from GenericFolderDataset import GenericFolderDataset  
# ----------------------------------------

try:
    from CAVE_dataset_2 import CAVEDataset
    from ARAD_dataset import TrainARADDataset, TestARADDataset
except ImportError:
    pass


# =======================================================================
# ALGORITMO SEI (Self-Evaluation-Index)
# =======================================================================
class self_evaluation_index(nn.Module):
    def __init__(self, msfa_size):
        super().__init__()
        self.msfa_size = msfa_size

    @staticmethod
    def pixel_shuffle_inv(tensor, scale_factor):
        '''
        Implementation of inverted pixel shuffle
        '''
        num, ch, height, width = tensor.shape
        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError('height and widht of tensor must be divisible by '
                             'scale_factor.')

        new_ch = ch * (scale_factor * scale_factor)
        new_height = height // scale_factor
        new_width = width // scale_factor

        shuffled_tensor = tensor.reshape(
            num, ch, new_height, scale_factor, new_width, scale_factor)
        shuffled_tensor = shuffled_tensor.permute(0, 1, 3, 5, 2, 4)
        shuffled_tensor = shuffled_tensor.reshape(num, new_ch, new_height, new_width)
        return shuffled_tensor

    def forward(self, reconstructed_cube):
        '''
        Calculating unsupervised metrics Self-Evaluation-Index (SEI)
        '''
        num_band = self.msfa_size ** 2
        # Device esplicito per evitare errori di runtime
        var_cube = torch.ones(1, num_band, device=reconstructed_cube.device)
        for i in range(num_band):
            singleband = reconstructed_cube[:, i, :, :].unsqueeze(1) 
            singleband_subimgs = self.pixel_shuffle_inv(singleband, scale_factor=self.msfa_size) 
            singleband_subimgs_avg = torch.mean(torch.mean(singleband_subimgs,-1), -1) 
            singleband_subimgs_var = singleband_subimgs_avg.var(dim=-1) 
            var_cube[0, i] = singleband_subimgs_var 
        SEI_cube = torch.mean(var_cube, dim=1)

        return SEI_cube


parser = argparse.ArgumentParser(description="Multispectral Image Demosaicing Toolbox")
parser.add_argument('--method', type=str, default='FGLN')
parser.add_argument('--msfa_size', type=int, default=3)
parser.add_argument('--dataset', type=str, default='Flavia', help='CAVE, ARAD, Flavia') 
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--end_epoch', type=int, default=4000, help="number of iterations")
parser.add_argument('--init_lr', type=float, default=2e-4, help="initial learning rate")
parser.add_argument('--lr_step', type=int, default=1000, help="adjust learning rate")
parser.add_argument('--outf', type=str, default='./train_exp/My/', help='output files')
parser.add_argument('--train_dir', type=str, default='/mnt/Volume_Interno/UBUNTU/dataset_flavia/train/')
parser.add_argument('--test_dir', type=str, default='/mnt/Volume_Interno/UBUNTU/dataset_flavia/valid/')
parser.add_argument('--cache', type=str, default='./training_cache/', help='training data cache')
parser.add_argument('--is_freq_loss', action='store_false') 
parser.add_argument('--gpu_id', type=str, default='0')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

print(f"Start training: {opt.method}")
print(f"\nLoading dataset: {opt.dataset}...")

cache_path = os.path.join(opt.cache, opt.dataset)
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
opt.cache_path = cache_path


if opt.dataset == 'CAVE':
    patch_size = 64
    stride = 32
    train_data = CAVEDataset(opt, type='train', patch_size=patch_size, stride=stride, augment=False)
    test_data = CAVEDataset(opt, type='test')
elif opt.dataset == 'ARAD':
    patch_size = 160
    train_data = TrainARADDataset(data_root=opt.train_dir, msfa_size=opt.msfa_size, patch_size=patch_size, augment=False)
    test_data = TestARADDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size)
elif opt.dataset == 'Flavia' or opt.dataset == 'Generic':
    patch_size = 60  
    if opt.msfa_size == 3:
        msfa_pattern = np.array([0, 5, 2, 3, 8, 7, 6, 1, 4])
    else:
        msfa_pattern = np.arange(opt.msfa_size ** 2).reshape(opt.msfa_size, opt.msfa_size)
        
    train_data = GenericFolderDataset(opt.train_dir, msfa_size=opt.msfa_size, msfa_pattern=msfa_pattern, 
                                      norm_flag=True, patch_size=patch_size, augment=False, type='train')
    test_data = GenericFolderDataset(opt.test_dir, msfa_size=opt.msfa_size, msfa_pattern=msfa_pattern, 
                                     norm_flag=True, patch_size=510, augment=False, type='test')

# Ottimizzazione DataLoader per saturazione GPU
train_loader = DataLoader(
    dataset=train_data, 
    batch_size=opt.batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=2
)

test_loader = DataLoader(
    dataset=test_data, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True, 
    persistent_workers=True
)

print(f'Training set samples: {len(train_data)}')
print(f'Test set samples: {len(test_data)}')
total_iteration = opt.end_epoch * len(train_loader)
print(f'Total iteration: {total_iteration}')

# Rimosse metriche inutilizzate: SSIM, SAM, ERGAS
criterion_psnr = Loss_PSNR()
criterion_sei = self_evaluation_index(msfa_size=opt.msfa_size)

criterion = (L1_Charbonnier_mean_loss(), reconstruction_loss(msfa_size=opt.msfa_size), FocalFrequencyLoss(msfa_size=opt.msfa_size))

pretrained_model_path = opt.pretrained_model_path
model = model_generator(opt.method, opt.msfa_size, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_psnr.cuda()
    criterion_sei.cuda() 
    criterion[0].cuda() 
    criterion[1].cuda() 
    criterion[2].cuda() 

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

def random_crop_4DTensor(a, crop_size):
    N, C, hei, wid=a.size()
    Height = random.randint(0, hei - crop_size)
    Width = random.randint(0, wid - crop_size)
    return a[:, :, Height:(Height + crop_size), Width:(Width + crop_size)]

def transform_opt(HR_4x, msfa_size, tran_type='random'):
    if tran_type == 'random':
        tran_type = random.choice([0, 1, 3, 5])
    elif tran_type == 'rotation': 
        tran_type = 0
    elif tran_type == 'flip': 
        tran_type = 1
    elif tran_type == 'resize': 
        tran_type = 3
    elif tran_type == 'patternshift': 
        tran_type = 5
    else:
        raise Exception("wrong tran type")

    if tran_type == 0:
        rn = random.randint(1, 3) 
        new_lable = torch.rot90(HR_4x, rn, [2, 3]) 

    if tran_type == 1:
        if np.random.uniform() < 0.5:
            new_lable = torch.flip(HR_4x, [2]) 
        else:
            new_lable = torch.flip(HR_4x, [3]) 

    scale_lib = [0.2, 0.25, 0.5, 2, 2.5, 3] 
    
    if tran_type == 3:
        while 1:
            scale_num = random.randint(0, len(scale_lib)-1)
            new_lable = torch.nn.functional.interpolate(HR_4x, scale_factor=scale_lib[scale_num], mode='bicubic')
            N, C, H, W = new_lable.size()
            if H >= (patch_size-opt.msfa_size)*scale_lib[0] and H <= patch_size*scale_lib[-1]:
                break
        new_lable = random_crop_4DTensor(new_lable, (H // opt.msfa_size) * opt.msfa_size) 

    if tran_type == 5:
        while 1:
            i = random.randint(0, msfa_size-1)
            j = random.randint(0, msfa_size-1)
            if i != 0 or j != 0:
                break
        new_lable = torch.roll(HR_4x, (-i, -j), (2, 3)) 
        N, C, H, W = new_lable.size()
        new_lable = new_lable[:, :, 0: (H - opt.msfa_size), 0: (W - opt.msfa_size)] 

    return new_lable

def get_sparsecube_raw(img_tensor, msfa_size):
    mask = torch.zeros_like(img_tensor)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1

    return mask.mul(img_tensor), torch.sum(mask.mul(img_tensor), 1).unsqueeze(1)

def adjust_learning_rate(optimizer, epoch, step=1000):
    lr = opt.init_lr * (0.5 ** (epoch // step))
    return lr

def train(opt, train_loader, optimizer, scheduler, method, model, criterion, epoch, num_epochs):
    if scheduler != None:
        lr = optimizer.param_groups[0]['lr']
    else:
        lr = adjust_learning_rate(optimizer, epoch - 1, opt.lr_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train_bar = tqdm(train_loader)
    model.train()
    batchSize_count = 0

    for batch in train_bar:
        # Trasferimento asincrono alla GPU senza Variable
        input_raw = batch[0].cuda(non_blocking=True)
        input_sparse_raw = batch[1].cuda(non_blocking=True)
        gt = batch[2].cuda(non_blocking=True)
        
        N, C, H, W = batch[0].size()
        batchSize_count += N

        firstcube = model(input_raw, input_sparse_raw)

        mosaic_loss = criterion[1](firstcube, gt)
        loss = mosaic_loss.clone()

        if opt.is_freq_loss:
            mosaic_loss_freq = criterion[2](firstcube, gt, need_mask=True, need_matrix=False,
                                            is_simulate=True)  
            loss += mosaic_loss_freq

        new_label = transform_opt(firstcube, opt.msfa_size)
        new_label = new_label.detach() 
        input_sparse_raw_new, input_raw_new = get_sparsecube_raw(new_label, opt.msfa_size)
        
        curoutput = model(input_raw_new, input_sparse_raw_new)
        cube_loss = criterion[0](curoutput, new_label)
        loss += cube_loss

        if opt.is_freq_loss:
            cube_loss_freq = criterion[2](curoutput, new_label, need_mask=False, need_matrix=False,
                                          is_simulate=True)  
            loss += cube_loss_freq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        if opt.is_freq_loss:
            train_bar.set_description(desc='Epoch[%d/%d] Mosaic_Loss: %.9f, Mosaic_Loss_Frequency: %.9f, Cube_Loss: %.9f, Cube_Loss_Frequency: %.9f, All_Loss: %.9f, lr: %.9f' %
                                       (epoch, num_epochs, mosaic_loss.item() / batchSize_count, mosaic_loss_freq.item() / batchSize_count, cube_loss.item() / batchSize_count,
                                        cube_loss_freq.item() / batchSize_count, loss.item() / batchSize_count, lr))
        else:
            train_bar.set_description(desc='Epoch[%d/%d] Mosaic_Loss: %.9f, Cube_Loss: %.9f, All_Loss: %.9f, lr: %.9f' %
                     (epoch, num_epochs, mosaic_loss.item() / batchSize_count, cube_loss.item() / batchSize_count, loss.item() / batchSize_count, lr))

    if opt.is_freq_loss:
        logger.info('Epoch[%d/%d] Mosaic_Loss: %.9f, Mosaic_Loss_Frequency: %.9f, Cube_Loss: %.9f, Cube_Loss_Frequency: %.9f, All_Loss: %.9f, lr: %.9f' %
            (epoch, num_epochs, mosaic_loss.item() / batchSize_count, mosaic_loss_freq.item() / batchSize_count, cube_loss.item() / batchSize_count,
             cube_loss_freq.item() / batchSize_count, loss.item() / batchSize_count, lr))
    else:
        logger.info('Epoch[%d/%d] Mosaic_Loss: %.9f, Cube_Loss: %.9f, All_Loss: %.9f, lr: %.9f' %
            (epoch, num_epochs, mosaic_loss.item() / batchSize_count, cube_loss.item() / batchSize_count, loss.item() / batchSize_count, lr))

def validate(opt, test_loader, method, model, epoch, num_epochs):
    test_bar = tqdm(test_loader)
    model.eval()
    metric_psnr = AverageMeter()
    metric_sei = AverageMeter() 

    with torch.no_grad():
        for batch in test_bar:
            # Trasferimento asincrono alla GPU senza Variable
            input_raw = batch[0].cuda(non_blocking=True)
            input_sparse_raw = batch[1].cuda(non_blocking=True)
            gt = batch[2].cuda(non_blocking=True)

            output = model(input_raw, input_sparse_raw)

            psnr = criterion_psnr(output, gt)
            sei_val = criterion_sei(output).mean().item()

            # Record metrics (solo PSNR e SEI)
            metric_psnr.update(psnr.data)
            metric_sei.update(sei_val) 

            # Formattazione aggiornata: PSNR standard e SEI in notazione scientifica
            test_bar.set_description(desc='Epoch[%d/%d] psnr: %.4f, sei: %.4e' %
                                          (epoch, num_epochs, metric_psnr.avg, metric_sei.avg))
    
    logger.info('Epoch[%d/%d] psnr: %.4f, sei: %.4e' %
                                          (epoch, num_epochs, metric_psnr.avg, metric_sei.avg))


def main():
    cudnn.benchmark = True
    for epoch in range(1, opt.end_epoch+1):
        train(opt, train_loader, optimizer, scheduler, opt.method, model, criterion, epoch, opt.end_epoch)
        
        # Validazione ed elaborazione metriche limitate ad ogni 10 epoche
        if epoch % 10 == 0:
            validate(opt, test_loader, opt.method, model, epoch, opt.end_epoch)
            print(f'Saving to {opt.outf}')
            save_checkpoint(opt.outf, epoch, model, optimizer)

if __name__ == '__main__':
    main()
    print(torch.__version__)