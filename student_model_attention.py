import torch
import torch.nn as nn
import math
import numpy as np

def get_WB_filter_msfa(msfa_size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    size = 2*msfa_size-1
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(msfa_size**2)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (msfa_size**2))
    filter0 = np.reshape(BilinearFilter, (size, size))
    return torch.from_numpy(filter0).float()

# ── NUOVO MODULO: Channel Attention (Squeeze-and-Excitation) ──
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        # Squeeze: comprime la risoluzione spaziale (H,W) a 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: apprende i pesi di calibrazione per i canali
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Calcola il vettore di attenzione
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Pesa i canali in input moltiplicandoli per l'attenzione
        return x * y.expand_as(x)


class _ResConvBlock(nn.Module):
    """Blocco convoluzionale residuale con PReLU e Channel Attention per la rete Student."""
    def __init__(self, channels):
        super(_ResConvBlock, self).__init__()
        self.conv1  = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.prelu1 = nn.PReLU(num_parameters=channels, init=0.02)
        
        self.conv2  = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.prelu2 = nn.PReLU(num_parameters=channels, init=0.02)
        
        self.conv3  = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        
        # Inserimento del modulo di attenzione
        self.ca = ChannelAttention(channels, reduction=4)

    def forward(self, x):
        a = self.prelu1(self.conv1(x))
        a = self.prelu2(self.conv2(a))
        a = self.conv3(a)
        
        # Calibrazione dei canali prima della skip connection
        a = self.ca(a)
        
        return x + a


class StudentDemosaic(nn.Module):
    def __init__(self, msfa_size=3, channels=16, num_blocks=2):
        super(StudentDemosaic, self).__init__()
        self.msfa_size    = msfa_size
        self.out_channels = msfa_size ** 2

        # ── Branch 1: Baseline bilineare congelata (identica al Teacher) ──
        self.WB_Conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=2 * msfa_size - 1,
            stride=1,
            padding=msfa_size - 1,
            bias=False,
            groups=self.out_channels,
        )

        # ── Branch 2: Rete residuale apprendibile ─────────────────────────
        self.conv_input = nn.Conv2d(
            self.out_channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu_in = nn.PReLU(num_parameters=channels, init=0.02)

        layers = []
        for _ in range(num_blocks):
            layers.append(_ResConvBlock(channels))
        self.deep_blocks = nn.Sequential(*layers)

        self.conv_tail = nn.Conv2d(
            channels, self.out_channels, kernel_size=3, padding=1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                if m is self.WB_Conv:
                    # Filtro bilineare congelato — identico al Teacher
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(self.msfa_size)
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                    for param in m.parameters():
                        param.requires_grad = False

                elif m is self.conv_tail:
                    # Pesi e bias a zero: all'epoca 0 il residuo è zero
                    # e l'output coincide esattamente con WB_Conv(x)
                    m.weight.data.zero_()
                    if m.bias is not None:
                        m.bias.data.zero_()

                else:
                    # Inizializzazione He standard per i layer intermedi
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        # 1. Baseline bilineare (congelata, non apprendibile)
        wb_norelu = self.WB_Conv(x)

        # 2. Residuo apprendibile
        out = self.relu_in(self.conv_input(wb_norelu))
        out = self.deep_blocks(out)
        out = self.conv_tail(out)

        # 3. Baseline + residuo
        return torch.add(wb_norelu, out)
    
if __name__ == '__main__':
    from torchinfo import summary
    model = StudentDemosaic(3, 32, 3)
    summary(model, input_size=(1, 9, 120, 120))