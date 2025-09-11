import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import toml
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from typing import Union, List
from einops import rearrange
from auxiliary_models.gtcrn import GTCRN
from auxiliary_models.NeXt_TDNN import NeXtTDNN
from auxiliary_models.TSConvNeXt import TSConvNeXt

class LayerNorm1d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-1, -2)
        return x
    
class embedoutput(nn.Module):
    def __init__(self, channel_size, embeding_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(channel_size, embeding_size * 2, kernel_size=1)
        self.gate_norm = LayerNorm1d(channel_size,eps=1e-6,affine=False)
        self.apply(self._init_weights)
        
    def forward(self, x):
        x_a, x_b = self.conv_1(x).chunk(2, dim=1)
        out = x_a * torch.sigmoid(x_b)
        return out

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)): # ⚡
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

class stftpreprocess(nn.Module):
    def __init__(self,                 
                n_fft: int,
                hop_length: int,
                win_length: int, 
                press,
                **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(self.win_length) # 256
        window = window / window.sum() 
        window = window.pow(0.5)         
        self.register_buffer('stft_window', window)
        
        self.press = press
        
    def forward(self, x, spec_type = "complex"):
        """
        :param x: [B, wave length]
        :return: [B, F, T] complex
        """
        pad = (self.n_fft - self.hop_length) // 2
        x = x.unsqueeze(1)  # [B, 1, wave_length]
        x = F.pad(x, (pad, pad), mode='reflect')
        x = x.squeeze(1)  # [B, wave_length + 2*pad]
        
        # stft
        spec_ori = torch.stft(x, 
                            n_fft = self.n_fft, 
                            hop_length = self.hop_length, 
                            win_length = self.win_length, 
                            window = self.stft_window, 
                            return_complex=True)
        
        # compress complex
        if spec_type == "complex":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5)) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex
            elif self.press == "None":
                spec = spec_ori
            else:
                spec = torch.pow(spec_ori.abs(), self.press) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex
        elif spec_type == "amplitude":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5))   # [B, F, T], complex
            elif self.press == "None":
                spec = spec_ori
            else:
                spec = torch.pow(spec_ori.abs(), self.press)  # [B, F, T], complex

        return spec

class stftpostprocess(nn.Module):
    def __init__(self,                 
                n_fft: int,
                hop_length: int,
                win_length: int, 
                press,
                **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.press = press

    def forward(self, x, length, mix_std = 1):
        if self.press == "None":
            spec = x  # [B, F, T], complex
        else:
            # reverse compression
            magnitude = x.abs()  # [B, F, T]
            phase = x.angle() 
            spec = torch.pow(magnitude + 1e-8, 1 / self.press) * torch.exp(1j * phase)  # [B, T, F]
        
        wav = torch.istft(spec, 
                            n_fft = self.n_fft, 
                            hop_length = self.hop_length, 
                            win_length = self.win_length, 
                            window=torch.hann_window(self.win_length).pow(0.5).to(x.device),
                            center=True,
                            length=length)
        wav = wav * mix_std

        return wav

class MelConvert(nn.Module):
    def __init__(self, n_fft: int, sr: int, n_mels: int, f_min: float=0.0, f_max: float=None, press: str="log"):
        super().__init__()
        self.n_fft = n_fft
        sr = 16000
        n_mels = 80  # 已经是整数

        f_min = 0.0  # float
        f_max = sr / 2.0  # 这会是 float

        self.press = press

        mel_fb = torchaudio.functional.melscale_fbanks(
            int(n_fft // 2 + 1), 
            float(f_min),  
            float(f_max),
            int(n_mels), 
            int(sr)
        )
        self.register_buffer('mel_fb', mel_fb)
    
    def forward(self, spec_complex):
        spec_amp = spec_complex.abs()  # [B, F, T]
        mel_spec = torch.matmul(self.mel_fb.transpose(-1, -2), spec_amp)  # [B, n_mels, T]
        if self.press == "log":
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        else:
            mel_spec = torch.pow(mel_spec, self.press)
        
        return mel_spec


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, embedding_dim, no_classes, scale = 30.0, margin=0.4):
        '''
        https://github.com/tomastokar/Additive-Margin-Softmax/blob/main/AMSloss.py
        Additive Margin Softmax Loss


        Attributes
        ----------
        embedding_dim : int 
            Dimension of the embedding vector
        no_classes : int
            Number of classes to be embedded
        scale : float
            Global scale factor
        margin : float
            Size of additive margin        
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.embedding = nn.Embedding(no_classes, embedding_dim, max_norm=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        '''
        Input shape (N, embedding_dim)
        '''
        n, m = x.shape        
        assert n == len(labels)
        assert m == self.embedding_dim
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.no_classes

        x = F.normalize(x, dim=1)
        w = self.embedding.weight        
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - self.margin
        
        onehot = F.one_hot(labels, self.no_classes)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)        
        err = self.loss(logits, labels)
        
        return err, logits


class spkembed_encoder(nn.Module):
    def __init__(self, 
                config,
                **kwargs):
        super().__init__()
        # preprocess
        self.stft_module = stftpreprocess(**config['FFT'], press = "None")
        self.mel_module = MelConvert(int(config['FFT']['n_fft']), 16000, 80, press="log")
        
        # fft
        n_fft = config['FFT']['n_fft']
        self.n_freqs = int(n_fft // 2 + 1) # 257

        # 1. denoiser
        self.denoiser = GTCRN()

        # spk embedding 
        self.spk_enc = NeXtTDNN(in_chans = 80,
                                depths=[1, 1, 1], 
                                dims=[256, 256, 256],
                                drop_path_rate=0.,
                                kernel_size = [7, 65],
                                block = "TSConvNeXt", # TSConvNeXt_light or TSConvNeXt
                                )

        # out 
        self.out_layer = embedoutput(channel_size = 256*3, embeding_size = 256)

        # istft
        self.istft = stftpostprocess(**config['FFT'], press = 'None')

        # loss
        self.spk_num = kwargs.get('spk_num')
        self.loss = AdMSoftmaxLoss(
            embedding_dim = 256, 
            no_classes = self.spk_num, # vctk 100
            scale = 40.0, 
            margin = 0.3
        )

    def compute_wave_loss(self, enhanced_wave, clean_wave, sample_rate):
        # SI-SNR
        def si_snr(est, ref, eps=1e-8):
            ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True)
            proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref / (ref_energy + eps)
            noise = est - proj
            ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
            return -10 * torch.log10(ratio + eps)

        # STFT
        n_fft = 512
        hop_length = 128
        win_length = 512
        window = torch.hann_window(win_length).to(enhanced_wave.device)

        enhanced_stft = torch.stft(enhanced_wave, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window, return_complex=True)
        clean_stft = torch.stft(clean_wave, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window, return_complex=True)

        # Magnitude Loss (MAE)
        # mag_loss = F.l1_loss(torch.abs(enhanced_stft), torch.abs(clean_stft))

        # Phase-Sensitive Loss
        phase_loss = F.mse_loss(enhanced_stft.real, clean_stft.real) + F.mse_loss(enhanced_stft.imag, clean_stft.imag)

        # SI-SNR Loss
        sisnr_loss = si_snr(enhanced_wave, clean_wave).mean()

        # Total Loss
        total_loss = sisnr_loss + phase_loss * 0.1

        return total_loss
    
    def forward(self, x, clean_sample, labels = None, training = False):
        # stft
        cplx_spec = self.stft_module(x, 'complex') # [batch, F, T], complex
        stft_specs = torch.stack([cplx_spec.real, cplx_spec.imag], dim = -1) # [batch, F, T, 2]

        # denoiser
        denoised_spec = self.denoiser(stft_specs) # [batch, F, T, 2]
        denoised_spec_cplx = torch.complex(denoised_spec[:,:,:,0],denoised_spec[:,:,:,1])

        # istft
        denoise_wave = self.istft(denoised_spec_cplx.squeeze(1), x.size(-1), 1)
        
        # mel
        mel = self.mel_module(denoised_spec_cplx) # [batch, n, T]
        
        # encoder
        spk_out = self.spk_enc(mel)
        
        # out
        embed_T = self.out_layer(spk_out)
        embed_mean = torch.mean(embed_T, dim = -1) # [batch, embed dim]
        
        # loss
        if training:
            wave_loss = self.compute_wave_loss(denoise_wave, clean_sample, sample_rate = 16000)
            loss = self.loss(embed_mean, labels)
            return embed_mean, wave_loss, loss
        if not training:
            return embed_mean, embed_T
        
        
