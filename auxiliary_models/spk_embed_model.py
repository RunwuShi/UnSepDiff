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
        window = window / window.sum()   # 归一化，使得窗口和为 1
        window = window.pow(0.5)           # 取平方根
        self.register_buffer('stft_window', window)
        
        self.press = press
        
    def forward(self, x, spec_type = "complex"):
        """
        :param x: [B, wave length]
        :return: [B, F, T] complex
        """
        # std
        # std
        # mix_std_ = torch.std(x, dim=1, keepdim=True)  # [B, 1]
        # x = x / mix_std_  # RMS normalization
        # max
        # 归一化：将音频幅值限制在 [-1, 1]
        # x = x / (x.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        
        # 填充：使用 reflect 填充，确保 STFT 能够处理边缘
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


class MelConvert(nn.Module):
    def __init__(self, n_fft: int, sr: int, n_mels: int, f_min: float=0.0, f_max: float=None, press: str="log"):
        """
        参数：
          n_fft: 与 STFT 模块中一致的 FFT 窗口大小
          sr: 采样率
          n_mels: mel 频谱的通道数
          f_min: mel 滤波器最低频率
          f_max: mel 滤波器最高频率，如果为 None 则默认为 sr/2
          press: 压缩方式，“log” 表示取对数，否则认为传入的是指数（如 0.5 表示平方根）
        """
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
        """
        输入：
          spec_complex: [B, F, T] complex tensor，来自 STFT 模块
        输出：
          mel_spec: [B, n_mels, T]，根据 press 参数压缩后的 mel 频谱
        """
        # 计算幅值谱
        spec_amp = spec_complex.abs()  # [B, F, T]
        
        # 将幅值谱转换到 mel 频谱：矩阵乘法将 [F, T] 映射为 [n_mels, T]
        mel_spec = torch.matmul(self.mel_fb.transpose(-1, -2), spec_amp)  # [B, n_mels, T]
        
        # 进行压缩处理
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

        # loss
        self.spk_num = kwargs.get('spk_num')
        self.loss = AdMSoftmaxLoss(
            embedding_dim = 256, 
            no_classes = self.spk_num, # vctk 100
            scale = 40.0, 
            margin = 0.3
        )

    def forward(self, x, labels = None, training = False):
        # stft
        cplx_spec = self.stft_module(x, 'complex') # [batch, F, T], complex
        stft_specs = torch.stack([cplx_spec.real, cplx_spec.imag], dim = -1) # [batch, F, T, 2]

        # mel
        mel = self.mel_module(cplx_spec) # [batch, n, T]
        
        # encoder
        spk_out = self.spk_enc(mel)
        
        # out
        embed_T = self.out_layer(spk_out)
        embed_mean = torch.mean(embed_T, dim = -1) # [batch, embed dim]
        
        # loss
        if training:
            loss = self.loss(embed_mean, labels)
            return embed_mean, loss
        if not training:
            return embed_mean, embed_T
        
        
if __name__ ==  '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config =  toml.load('/misc/export3/shirunwu/work/diff_infer/config/spkencoder/spkencoder.toml')


    model = spkembed_encoder(config = config, spk_num = 100).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params',total_params)

    wave = torch.randn(64, 16000).to(device)
    label = torch.randint(0, 100, (64,)).to(device)
    out = model(wave, label, training = True)
    
    print(out)