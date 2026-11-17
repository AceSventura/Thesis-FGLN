import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class MACA(nn.Module):
    def __init__(self, msfa_size, channel, k_size=5, reduction=16, lambda_order=[0, 7, 4, 2, 3, 8, 1, 6, 5]):
        super().__init__()
        self.msfa_size = msfa_size
        self.channel = channel
        self.k_size = k_size

        # Gestione dell'ordinamento spettrale per lunghezze d'onda
        if lambda_order is not None:
            # Sostituiamo register_buffer per spostare automaticamente i vettori su GPU con il modello
            self.register_buffer('lambda_order', torch.tensor(lambda_order, dtype=torch.long))
            # Calcolo dell'indice inverso per ripristinare l'ordine geometrico nativo
            self.register_buffer('inverse_lambda_order', torch.argsort(self.lambda_order))
        else:
            self.lambda_order = None
            self.inverse_lambda_order = None

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=k_size, 
            padding=(k_size - 1) // 2, 
            bias=False
        )
        self.act1 = nn.Sigmoid()

        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        
        mid_channels = max(1, channel // reduction)
        self.fc2 = nn.Sequential(
            nn.Linear(channel, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Args:
            x: [b, c, h, w]
        Returns:
            out: [b, c, h, w]
        '''
        batch, C, H, W = x.size()
        
        x_in = x.view(-1, 1, H, W) # [b*c, 1, h, w]
        
        px = F.pixel_unshuffle(x_in, downscale_factor=self.msfa_size) # [b*c, msfa_size**2, h//msfa_size, w//msfa_size]
        py = self.avg_pool1(px).squeeze(-1).transpose(-2, -1)         # [b*c, 1, msfa_size**2]

        # =========================================================================
        # 1. PERMUTAZIONE SPETTRALE (PRIMA DELLA CONVOLUZIONE 1D)
        # =========================================================================
        if self.lambda_order is not None:
            py = py[:, :, self.lambda_order]

        # Convoluzione 1D lungo i vicini spettrali contigui
        py = self.conv1(py)

        # =========================================================================
        # 2. RIPRISTINO DELL'ORDINE GEOMETRICO (DOPO LA CONVOLUZIONE 1D)
        # =========================================================================
        if self.inverse_lambda_order is not None:
            py = py[:, :, self.inverse_lambda_order] 

        # =========================================================================
        py = py.transpose(-2, -1).unsqueeze(-1) # [b*c, msfa_size**2, 1, 1]
        py = self.act1(py)
        
        # Espansione spaziale e ripristino risoluzione originale
        py = py.expand_as(px)
        py = F.pixel_shuffle(py, upscale_factor=self.msfa_size) # [b*c, 1, h, w]
        
        pout = x_in * py
        pout = pout.view(batch, C, H, W)
        
        y = self.avg_pool2(x).view(batch, C)
        y = self.fc2(y).view(batch, C, 1, 1)
        
        # Sfruttamento del broadcasting nativo (rimosso y.expand_as)
        out = pout * y
        
        return out


if __name__ == '__main__':
    b, c, h, w = 1, 64, 480, 512
    inp = torch.randn(b, c, h, w)
    
    # Esempio di indici ordinati per lunghezza d'onda (per griglia 4x4)
    # Es. La posizione (0,0) corrisponde all'indice geometrico 0, ma potrebbe essere la 3ª banda più corta
    
    attn = MACA(msfa_size=3, channel=c, k_size=5, reduction=3, lambda_order=None)
    summary(attn, input_data=inp)