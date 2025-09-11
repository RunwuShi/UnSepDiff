import os
import toml
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from typing import Union, List
from einops import rearrange
from auxiliary_models.gtcrn import GTCRN
from auxiliary_models.NeXt_TDNN import NeXtTDNN
from auxiliary_models.TSConvNeXt import TSConvNeXt
from auxiliary_models.vap_tanh_fc_bn import VAP_BN_FC_BN

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
        x = x / (x.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        
        # 填充：使用 reflect 填充，确保 STFT 能够处理边缘
        pad = (self.n_fft - self.hop_length) // 2
        # 这里使用 F.pad 对每个样本进行 padding，输入需加上 channel 维度
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

class MelPreprocess(nn.Module):
    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 n_mels: int,
                 sr: int = 16000,
                 f_min: float = 0.0,
                 f_max: float = None,
                 press: str = "log",
                 **kwargs):
        """
        参数：
          n_fft: FFT 的窗口大小
          hop_length: 帧移
          win_length: 窗口长度
          n_mels: mel 频谱的通道数
          sr: 采样率
          f_min: mel 滤波器最低频率
          f_max: mel 滤波器最高频率，如果为 None 则默认为 sr/2
          press: 压缩方式，"log" 表示取对数，否则认为传入的是一个指数（如 0.5 表示平方根）
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sr = sr
        self.press = press

        # 生成窗函数，归一化后取平方根
        window = torch.hann_window(self.win_length)
        window = window / window.sum()  
        window = window.pow(0.5)
        self.register_buffer('stft_window', window)
        
        # 设置最高频率
        if f_max is None:
            f_max = self.sr / 2
        self.f_min = f_min
        self.f_max = f_max
        
        # 生成 mel 滤波器组
        # torchaudio.functional.melscale_fbanks 的输出 shape 为 [n_mels, n_fft//2+1]
        mel_fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, self.sr, n_mels, f_min, f_max)
        self.register_buffer('mel_fb', mel_fb)

    def forward(self, x, spec_type="amplitude"):
        """
        输入：
          x: [B, wave_length]，一维信号，未必归一化
        输出：
          mel_spec: [B, n_mels, T]，mel频谱图
        """
        # 归一化：将信号幅值归一化到 [-1, 1]
        x = x / (x.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        
        # 反射填充，保证边缘效果
        pad = (self.n_fft - self.hop_length) // 2
        x = x.unsqueeze(1)  # [B, 1, wave_length]
        x = F.pad(x, (pad, pad), mode='reflect')
        x = x.squeeze(1)  # [B, wave_length + 2*pad]
        
        # 计算 STFT，返回复杂谱
        spec = torch.stft(x,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.stft_window,
                          return_complex=True)  # [B, F, T]
        
        # 获取幅值谱
        spec_amp = spec.abs()  # [B, F, T]
        
        # 将幅值谱映射到 mel 频谱：mel_fb: [n_mels, F]
        # 使用矩阵乘法：将每个样本 [F, T] 映射为 [n_mels, T]
        mel_spec = torch.matmul(self.mel_fb, spec_amp)  # [B, n_mels, T]
        
        # 压缩幅值
        if spec_type == "amplitude":
            if self.press == "log":
                mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
            else:
                mel_spec = torch.pow(mel_spec, self.press)
        
        return mel_spec


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



class mix_src_encoder(nn.Module):
    def __init__(self, 
                h_channels,
                config,
                **kwargs):
        super().__init__()

        # preprocess
        self.stft_module = stftpreprocess(**config['FFT'], press = "None")
        self.mel_module = MelConvert(int(config['FFT']['n_fft']), 16000, 80, press="log")
        # self.preprocess = MelPreprocess(**config['FFT'], n_mels=80, sr=16000, press="log")

        # fft
        n_fft = config['FFT']['n_fft']
        self.n_freqs = int(n_fft // 2 + 1) # 257

        # 1. denoiser
        self.denoiser = GTCRN()

        # 2. spk embedding 
        self.spk_enc = NeXtTDNN(in_chans = self.n_freqs + 80 + 80,
                                depths=[1, 1, 1], 
                                dims=[256, 256, 256],
                                drop_path_rate=0.,
                                kernel_size = [7, 65],
                                block = "TSConvNeXt", # TSConvNeXt_light or TSConvNeXt
                                )

        # 3. final out
        self.aggregate = VAP_BN_FC_BN(
            channel_size = 3 * 256,
            intermediate_size = int(3 * (256/8)),
            embeding_size = 192
        )
        self.final_out = nn.Linear(192, 1)


    def forward(self, x):
        # stft
        cplx_spec = self.stft_module(x, 'complex') # [batch, F, T], complex
        stft_specs = torch.stack([cplx_spec.real, cplx_spec.imag], dim = -1) # [batch, F, T, 2]

        # 1. mel
        mel_1 = self.mel_module(cplx_spec) # [batch, n, T]

        # denoiser
        denoised_spec = self.denoiser(stft_specs) # [batch, F, T, 2]
        denoised_spec_cplx = torch.complex(denoised_spec[:,:,:,0],denoised_spec[:,:,:,1])
        denoised_spec_abs = torch.abs(denoised_spec_cplx) # [batch, F, T]
        
        # 2. mel
        mel_2 = self.mel_module(denoised_spec_cplx) # [batch, n, T]

        # process
        spk_in = torch.concat([mel_1, mel_2, denoised_spec_abs], dim = 1)
        spk_out = self.spk_enc(spk_in)
        
        # out result
        spk_embedding = self.aggregate(spk_out)
        out = self.final_out(spk_embedding).squeeze(-1) # [B]

        return out


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

class spkemb_encoder(nn.Module):
    def __init__(self, 
                h_channels,
                config,
                **kwargs):
        super().__init__()

        # preprocess
        self.stft_module = stftpreprocess(**config['FFT'], press = "None")
        self.mel_module = MelConvert(int(config['FFT']['n_fft']), 16000, 80, press="log")
        # self.preprocess = MelPreprocess(**config['FFT'], n_mels=80, sr=16000, press="log")

        # fft
        n_fft = config['FFT']['n_fft']
        self.n_freqs = int(n_fft // 2 + 1) # 257

        # 1. denoiser
        self.denoiser = GTCRN()

        # 2. spk embedding 
        self.spk_enc = NeXtTDNN(in_chans = self.n_freqs + 80 + 80,
                                depths=[3, 3, 3], 
                                dims=[256, 256, 256],
                                drop_path_rate=0.,
                                kernel_size = [7, 65],
                                block = "TSConvNeXt", # TSConvNeXt_light or TSConvNeXt
                                )

        # 3. final out
        self.aggregate = VAP_BN_FC_BN(
            channel_size = 3 * 256,
            intermediate_size = int(3 * (256/8)),
            embeding_size = 192
        )
        self.spk_num = kwargs.get('spk_num')

        # loss
        self.loss = AdMSoftmaxLoss(
            embedding_dim = 192, 
            no_classes = self.spk_num, # vctk 100
            scale = 40.0, 
            margin = 0.3
        )


    def forward(self, x, labels, training = False):
        # stft
        cplx_spec = self.stft_module(x, 'complex') # [batch, F, T], complex
        stft_specs = torch.stack([cplx_spec.real, cplx_spec.imag], dim = -1) # [batch, F, T, 2]

        # 1. mel
        mel_1 = self.mel_module(cplx_spec) # [batch, n, T]

        # denoiser
        denoised_spec = self.denoiser(stft_specs) # [batch, F, T, 2]
        denoised_spec_cplx = torch.complex(denoised_spec[:,:,:,0],denoised_spec[:,:,:,1])
        denoised_spec_abs = torch.abs(denoised_spec_cplx) # [batch, F, T]
        
        # 2. mel
        mel_2 = self.mel_module(denoised_spec_cplx) # [batch, n, T]

        # process
        spk_in = torch.concat([mel_1, mel_2, denoised_spec_abs], dim = 1) # [B, F', T]
        spk_out = self.spk_enc(spk_in) # [B, F', T]
        
        # out result
        spk_embedding = self.aggregate(spk_out) # [B, emb dim]

        # loss
        if training:
            loss = self.loss(spk_embedding, labels)
            return spk_embedding, loss
        if not training:
            return spk_embedding, spk_out


if __name__ ==  '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = toml.load('/gs/bs/tga-nakadailab/shirunwu/work/diff_infer/config/spkencoder/spkencoder.toml')

    # model = mix_src_encoder(256, config).to(device)
    model = spkemb_encoder(h_channels = 256, config = config, spk_num = 100).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params',total_params)

    wave = torch.randn(64, 48000).to(device)
    out = model(wave)

    a = 1


