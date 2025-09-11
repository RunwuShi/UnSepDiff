import enum
import torch 
import numpy as np
import torch.nn as nn
import toml

from tqdm.auto import tqdm
from torch.nn import functional as F
# from auxiliary_models.enhan_mix_src_model import *
# from auxiliary_models.spk_embed_model import *
from auxiliary_models.spk_embed_model_denoise import *



def enforce_zero_terminal_snr(gammas):
    """Shift the noise schedule in order to achieve the zero-SNR at very first step
    Please refer to https://arxiv.org/abs/2305.08891

    This code corresponds to Equation 8, and Fig.4 in our paper

    Args:
        gammas (torch.Tensor): Into gamma function

    Returns:
        torch.Tensor: zero-snr noise scheduler
    """
    alphas = 1 - gammas
    alphas_bar = alphas.cumprod(dim=0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone() - 1e-4

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= (alphas_bar_sqrt_0) / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    gammas = 1 - alphas
    return gammas
   
def get_noise_schedule(schedule_name, num_diffusion_timesteps, beta_start = 0.0001, beta_end = 0.02):
    if schedule_name == "linear":
        # schedule = np.linspace(
        #             beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        #         )
        
        schedule = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
        # cancel snr
        # schedule = enforce_zero_terminal_snr(schedule) # cancel snr
        schedule = schedule.numpy()
        return schedule


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:
        res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def load_spk_model(config_path: str, model_filename: str, device: torch.device = None):
    if device is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = toml.load(config_path)
    spk_model = mix_src_encoder(256, config).to(device)
    spk_model.load_state_dict(torch.load(model_filename, map_location=device))
    spk_model.eval()
    
    print(f"Speaker encoder loaded from {model_filename} on {device}")
    return spk_model


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def adjust_waveform_length_math(x: torch.Tensor, n_fft: int, hop_length: int, multiple: int = 8) -> torch.Tensor:
    if x.ndim == 3:
        x = x.squeeze(1)
    B, T = x.shape
    pad = n_fft // 2
    F_orig = (T + 2 * pad - n_fft) // hop_length + 1
    remainder = F_orig % multiple
    if remainder == 0:
        T_new = T
    else:
        F_target = F_orig - remainder
        T_new = F_target * hop_length - 2 * pad + n_fft - 1
    return x[:, :T_new]


def log_sum_exp(a, b, k):
    """
    Numerically stable Log-Sum-Exp function for smooth_max.
    Calculates log(exp(k*a) + exp(k*b)) / k
    """
    # Shift by the max value to prevent overflow
    max_val = np.maximum(k * a, k * b)
    return (max_val + np.log(np.exp(k * a - max_val) + np.exp(k * b - max_val))) / k

def smooth_max(a, b, k):
    """
    A smooth approximation of max(a, b) using the Log-Sum-Exp trick.
    This creates a "smooth floor".
    """
    return log_sum_exp(a, b, k)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    L1 = enum.auto()
    
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL



class GaussianDiffusion_spk(nn.Module):
    def __init__(
        self,
        steps = 1000, # 1000
        noise_schedule = "linear",
        model_mean_type = None,
        model_var_type = None,
        loss_type = None,
        **kwargs):
        super().__init__()

        beta_start = kwargs.get('beta_start', 0.0001)
        beta_end = kwargs.get('beta_end', 0.02)
        self.betas = get_noise_schedule(schedule_name = noise_schedule, num_diffusion_timesteps = steps, beta_start = beta_start, beta_end = beta_end)
        self.num_timesteps = int(self.betas.shape[0])
        
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        # config
        self.config = kwargs.get('config_file')
        
        # paras
        self.clip_denoised = True
        self.input_sigma_t = False
        self.rescale_timesteps = False # True

        alphas = 1.0 - self.betas
        self.alphas = alphas 
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)) 

        # smooth_max schedule
        target_floor = 0.002 
        sharpness_k = 1000.0  
        # self.smooth_max_schedule = smooth_max(self.posterior_variance, target_floor, k=sharpness_k)
        self.posterior_std_dev = 0.09 * np.sqrt(self.posterior_variance) # 0.09
        self.smooth_max_schedule = smooth_max(self.posterior_std_dev, target_floor, k=sharpness_k)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # load speaker model
        device = kwargs.get('device')
        spk_config = toml.load(self.config['spk_model_config_path'])
        
        # noisy version
        spk_ckpt_path = self.config['spk_model_ckpt_path']
        
        self.spk_model = spkembed_encoder(config = spk_config, spk_num = 100).to(device)
        self.spk_model.load_state_dict(torch.load(spk_ckpt_path, map_location=device, weights_only=True))
    
        for p in self.spk_model.parameters():
            p.requires_grad_(False)
        self.spk_model.train()
        


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self,
        model,
        x,
        t,
        clip_denoised,
        model_kwargs = None):
        """
        get p(x_{t-1} | x_t)
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, _ = x.shape[:2]
        assert t.shape == (B,)
        out = {}
        
        # model -> noise
        if self.config['t_mode'] == "sigma":
            model_output = model(x, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs).squeeze(1)
            model_output = model_output.unsqueeze(1)
            # if x.ndim == 3:
            #     x = x.squeeze(1)
        else:
            model_out = model(x.squeeze(1), self._scale_timesteps(t, True), **model_kwargs) 
            if isinstance(model_out, torch.Tensor):
                model_output = model_out.unsqueeze(1)
            elif isinstance(model_out, tuple):
                model_output = model_out[0].unsqueeze(1)
                spk_infos = model_out[1]
                out["spk_info"] = spk_infos

        # variance
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        # to tensor
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        # predict EPSILON
        def process_xstart(x):
            if self.clip_denoised:
                return x.clamp(-1, 1)
            return x

        # predict x(0)
        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )

        # p(x(t-1)|x(t),x(0))
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        
        out["mean"] = model_mean
        out["variance"] = model_variance
        out["log_variance"] = model_log_variance
        out["pred_xstart"] = pred_xstart

        return out

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        带噪 x(t) - noise ϵ -> x(0) 
        """
        assert x_t.shape == eps.shape, f"Shape mismatch: x_t={x_t.shape}, eps={eps.shape}"
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


    def p_sample(
        self,
        model,
        x,
        t,
        model_kwargs = None,
        prior = None):

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised = self.clip_denoised,
            model_kwargs = model_kwargs
        )  

        noise = torch.randn_like(x)
        if prior is not None:
            noise = noise * prior[None, None, ...]
        
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        # sample = out["mean"]
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"], "total_out": out}


    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        coef2 = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return coef1 * x_start + coef2 * noise


    def direct_sample(
        self, 
        model,
        shape,
        measurement, 
        n_src, 
        clip_denoised,
        degradation,
        model_kwargs = None,
        device = None,
        diffwave = False):

        if device is None:
            device = next(model.parameters()).device

        # bar
        pbar = tqdm(list(range(self.num_timesteps))[::-1], ncols=80)

        # noise
        noise_list = [torch.randn(*shape, device=device) for _ in range(n_src)]
        x = torch.cat(noise_list, dim=-1) 
        x.requires_grad_(False)
        
        snr = 0.00001
        xi = None

        corrector = CorrectorVPConditional(
            degradation=degradation,
            n_src = n_src,
            snr=snr,
            xi=xi,
            sde=self,
            score_fn=model,
            device=device,
        )

        for i in pbar:
            t = torch.tensor([i] * shape[0], device=device) 
            
            with torch.no_grad(): # unconditional: 
                # single src
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) # [single src,...]
                x_stack = torch.cat(x_solo, dim=0)
                t = torch.tensor([i] * x_stack.size(0), device=device) 
                
                if model_kwargs is not None:
                    label = torch.tensor(model_kwargs, dtype=torch.long).to(x_stack.device)
                    model_kwargs_in = {"condition": label}
                else:
                    model_kwargs_in = {}
                
                out = self.p_sample(model,x_stack,t,model_kwargs_in)
                
                # to one
                out_samples = out['sample'].reshape(1, 1, -1)
                out_means = out['mean'].reshape(1, 1, -1)   
                out_pred_xstart = out['pred_xstart'].reshape(1, 1, -1)  
                    
            y = measurement
            t = torch.tensor([i] * shape[0], device=device)

            # update 1
            coefficient = 0.5 # 0.5
            total_log_sum = 0
            steps = 2
            for i in range(steps):
                new_samples = []
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)
                
                # log p(y | x)
                log_p_y_x = y - x_sum
                total_log_sum += log_p_y_x
                
                start = 0
                end = y.size(-1)
                while end <= x.size(-1):
                    new_sample = (
                        out_samples[:, :, start:end] + coefficient * total_log_sum
                    )
                    new_samples.append(new_sample)
                    start = end
                    end += y.size(-1)
                x = torch.cat(new_samples, dim=-1)

            # update 2
            threshold = 150
            if t[0] < threshold and t[0] > 0:
                # to batch
                x = x.squeeze(1)
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) # [single src,...]
                x_stack = torch.cat(x_solo, dim=0)
                t_b = torch.tensor([t] * x_stack.size(0), device=x_stack.device) 
                
                
                if model_kwargs is not None:    
                    label = torch.tensor(model_kwargs, dtype=torch.long).to(measurement.device)
                    model_kwargs_in = {"condition": label}
                else:
                    model_kwargs_in = {}
                    
                # get noise
                if diffwave:
                    x_in = x_stack.unsqueeze(1)
                    eps = model(x_in, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs_in)
                    eps = eps.transpose(0, 1).squeeze(0) 
                else:
                    eps = model(x_stack, self._scale_timesteps(t_b, True), **model_kwargs_in)
                eps = eps.reshape(1, 1, -1)
                x = x.unsqueeze(1)
                
                # condition
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)
                condition = y - (x_sum) 

                # corrector
                x = corrector.langevin_corrector_sliced(x, t, eps, y, condition)
                
            out["sample"] = x.reshape(1, 1, -1)
            
            yield out
            x = out["sample"]

        return out


    def dirac_sample_eval(
        self, 
        model,
        shape, 
        measurement, 
        n_src, 
        degradation, 
        model_kwargs = None,
        device = None,
        diffwave = False):

        if device is None:
            device = next(model.parameters()).device

        # noise
        noise = torch.randn(1, n_src, measurement.size(-1)).to(device)

        # guidance scale
        posterior_std_dev = np.flip(self.posterior_std_dev)

        # Set initial noise
        s_churn = 40 
        sigmas = self.sqrt_one_minus_alphas_cumprod
        sigmas_denoising_order = np.flip(sigmas, axis=0) 
        simga_0 = sigmas[0] 
        x = simga_0 * noise 
    

        for i in tqdm(range(len(sigmas_denoising_order) - 1), ncols=80): 
            sigma = sigmas_denoising_order[i]
            sigma_next = sigmas_denoising_order[i+1]
            t_idx = self.num_timesteps - 1 - i 
            if t_idx < 0:
                t_idx = 0 
            num_resamples = 2 
            for r in range(num_resamples): 
                gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
                sigma_hat = sigma * (gamma + 1) 
                
                noise_scale = (sigma_hat ** 2 - sigma ** 2) ** 0.5
                x = x + torch.randn_like(x) * noise_scale
                x = x.to(measurement.device)
                
                t = torch.tensor([t_idx] * n_src, device=measurement.device, dtype=torch.long)
                # print('t',t) 

                source_id = 1 
                x[:, [source_id], :] = measurement - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
                
                # to model
                if model_kwargs is not None:
                    label = torch.tensor(model_kwargs, dtype=torch.long).to(measurement.device)
                    model_kwargs_in = {"condition": label}
                else: 
                    model_kwargs_in = {} 
                
                if diffwave:
                    x_in = x.transpose(0, 1)
                    model_output = model(x_in, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs_in)
                    predicted_noise = model_output
                    predicted_noise = predicted_noise.transpose(0, 1).squeeze(0)  
                else:  
                    x_in = x.squeeze(0)
                    predicted_noise = model(x_in, self._scale_timesteps(t, True), **model_kwargs_in)
                    
                score = predicted_noise
                
                # guidance start
                # x(0)
                def process_xstart(x):
                    if self.clip_denoised:
                        return x.clamp(-1, 1)
                    return x
                
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_in, t=t, eps=score)
                ) # x(0)

                pred_xstart_detached = pred_xstart.detach().clone().requires_grad_(True)  
                x_in_detached = x_in.detach().clone().requires_grad_(True)  
                
                # guidance over
                scores = [score[si,:] for si in range(n_src)]
                ds = [s - score[source_id,:] for s in scores]
                d_srcs = torch.stack(ds, dim=0)
                x = x + d_srcs.unsqueeze(0) * (sigma_next - sigma_hat)
                
                # Renoise if not last resample step
                if r < num_resamples - 1:
                    x = x + np.sqrt(sigma ** 2 - sigma_next ** 2) * torch.randn_like(x)
                    
        out = {}
        out["sample"] = x.reshape(1, 1, -1) 

        yield out
        return out



    def dirac_sample_dps_eval(
        self, 
        model,
        shape, 
        measurement, 
        n_src, 
        degradation, 
        model_kwargs = None,
        device = None,
        diffwave = False):

        if device is None:
            device = next(model.parameters()).device

        # for dps corrector
        snr = 0.00001
        xi = None
        corrector = CorrectorVPConditional(
            degradation=degradation,
            n_src = n_src,
            snr=snr,
            xi=xi,
            sde=self,
            score_fn=model,
            device=device,
        )

        # guidance scale
        posterior_std_dev = np.flip(self.posterior_std_dev)
        
        # noise
        noise = torch.randn(1, n_src, measurement.size(-1)).to(device)
        
        # Set initial noise
        s_churn = 40 
        sigmas = self.sqrt_one_minus_alphas_cumprod
        sigmas_denoising_order = np.flip(sigmas, axis=0) 
        simga_0 = sigmas[0]
        x = simga_0 * noise 

        # 1. dirac sampling
        dirac_round = 200 - 1
        for i in tqdm(range(0, dirac_round, 1), ncols=80):
            sigma = sigmas_denoising_order[i]
            sigma_next = sigmas_denoising_order[i+1]
            t_idx = self.num_timesteps - 1 - i 
            if t_idx < 0:
                t_idx = 0 

            num_resamples = 2
            for r in range(num_resamples): 
                gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
                sigma_hat = sigma * (gamma + 1) 
                
                noise_scale = (sigma_hat ** 2 - sigma ** 2) ** 0.5
                x = x + torch.randn_like(x) * noise_scale
                x = x.to(measurement.device)
                t = torch.tensor([t_idx] * n_src, device=measurement.device, dtype=torch.long)
                # print('t',t) 

                source_id = 0 
                x[:, [source_id], :] = measurement - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
                
                # to model
                if model_kwargs is not None:
                    label = torch.tensor(model_kwargs, dtype=torch.long).to(measurement.device)
                    model_kwargs_in = {"condition": label}
                else: 
                    model_kwargs_in = {} 
                
                if diffwave:
                    x_in = x.transpose(0, 1)
                    model_output = model(x_in, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs_in)
                    predicted_noise = model_output
                    predicted_noise = predicted_noise.transpose(0, 1).squeeze(0)  
                else:  
                    x_in = x.squeeze(0)
                    predicted_noise = model(x_in, self._scale_timesteps(t, True), **model_kwargs_in)
                    
                score = predicted_noise
                
                # guidance start
                # x(0)
                def process_xstart(x):
                    if self.clip_denoised:
                        return x.clamp(-1, 1)
                    return x
                
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_in.squeeze(1), t=t, eps=score)
                ) # x(0)

                pred_xstart_detached = pred_xstart.detach().clone().requires_grad_(True)  
                x_in_detached = x_in.detach().clone().requires_grad_(True)  
                
                # spk embedding guidance
                if 75 <= t_idx <= 175: 
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        embed_mean, embed_T = self.spk_model(x_in_detached, None) # x_in_detached, pred_xstart_detached
                    embed_T_1 = embed_T[0]   # shape: [D, T1]
                    embed_T_2 = embed_T[1]   # shape: [D, T2]

                    all_spk = torch.cat([embed_T_1, embed_T_2], dim=-1)  # [D, T1+T2]
                    emb = all_spk.T  # [T1+T2, D]
                    emb_norm = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)

                    att_cosine = emb_norm @ emb_norm.T
                    T = att_cosine.size(0) // 2 
                    att_cos_src1 = att_cosine[:T, :T]   # [T, T]
                    att_cos_src2 = att_cosine[T:, T:]   # [T, T]
                    
                    loss_src1 = (1.0 - att_cos_src1).mean()
                    loss_src2 = (1.0 - att_cos_src2).mean()

                    # cross
                    att_cross = att_cosine[:T, T:]  # [T1, T2]
                    loss_cross = att_cross.mean()
                    p1, p2, p3 = 10.0, 10.0, 10.0
                    
                    loss_spk_embed = p1 * loss_src1 + p2 * loss_src2 + p3 * loss_cross


                    grad = torch.autograd.grad(loss_spk_embed,
                                                x_in_detached, # x_in_detached, pred_xstart_detached
                                                retain_graph=False,
                                                create_graph=False)[0] 
    
                    # cut
                    grad_max_norm = 1.0 
                    torch.nn.utils.clip_grad_norm_([grad], grad_max_norm)    

                    # update
                    norm_dims = list(range(1, grad.ndim))
                    grad_norm = torch.linalg.norm(grad, dim=norm_dims, keepdim=True)
                    _, h = score.shape
                    r_up = torch.sqrt(torch.tensor(h)) * self.posterior_std_dev[t_idx]
                    
                    # score update
                    score = score - r_up * (grad / (grad_norm + 1e-8))
                    # guidance over
                
                scores = [score[si,:] for si in range(n_src)]
                ds = [s - score[source_id,:] for s in scores]
                d_srcs = torch.stack(ds, dim=0)
                x = x + d_srcs.unsqueeze(0) * (sigma_next - sigma_hat)
                
                # Renoise if not last resample step
                if r < num_resamples - 1:
                    x = x + np.sqrt(sigma ** 2 - sigma_next ** 2) * torch.randn_like(x)


        # 2. dps/dsg sampling
        dps_step = 1 # 1
        pbar = tqdm(range(dps_step), ncols=80)

        x_conti = x.clone()
        for i in pbar:
            x_conti = x_conti.requires_grad_() 
            
            # conti
            x_conti = x_conti.reshape(1, 1, -1)
            
            # p sample for x
            x_solo = torch.split(x_conti, x_conti.size(-1) // n_src, dim=-1)
            x_stack = torch.cat(x_solo, dim=0)
            model_kwargs_in = None
            t = torch.tensor([i] * x_stack.size(0), device=device) 
            out = self.p_sample(model,x_stack,t,model_kwargs_in)
            
            # t time step
            t = torch.tensor([i], device=device) # size [1]

            # out sample
            out_samples = out['sample'].reshape(1, 1, -1)
            out_means = out['mean'].reshape(1, 1, -1)   
            out_pred_xstart = out['pred_xstart'].reshape(1, 1, -1)           
            
            # corrector
            x_conti, distance = corrector.update_fn_recons(
                                n_src,
                                x_t = out_samples, # update this
                                x_t_mean = out_means, # no use
                                measurement = measurement,
                                noisy_measurement = measurement, # no use
                                x_prev  = x_conti,
                                x_0_hat = out_pred_xstart,
                                time = t,
                                total_out = out)
            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            x_conti = x_conti.detach()
        
        x = x_conti
        out = {}
        out["sample"] = x.reshape(1, 1, -1) 
        
        yield out
        return out

    def _scale_timesteps(self, t, rescale_timesteps = False):
        if rescale_timesteps:
            t_scaled = (t.float() * (1000.0 / self.num_timesteps)).long()
            return t_scaled
        return t


class CorrectorVPConditional():
    def __init__(self, degradation, n_src, xi, sde, snr, score_fn, device):
        self.degradation = degradation
        self.n_src = n_src
        self.xi = xi
        self.alphas = torch.from_numpy(sde.alphas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) 
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.recip_alphas = torch.from_numpy(1 / sde.sqrt_one_minus_alphas_cumprod).to(
            device
        )
        self.device = device
        self.save_file = []
        
        len_src = 64000
        self.source_m = [torch.zeros(1, 1, len_src, device=self.device) for _ in range(self.n_src)]
        self.source_v = [torch.zeros(1, 1, len_src, device=self.device) for _ in range(self.n_src)]
        self.diffusion_step_adam = 0
        

    def update_fn_recons(self, n_src, x_t, x_t_mean, x_prev, x_0_hat, measurement, **kwargs):
        save_root = kwargs.get('save_path')
        
        t = kwargs.get('time')
        lamda = torch.sqrt(1 - self.alphas[t]).float()
        
        sigmas = self.sde.sqrt_one_minus_alphas_cumprod
        sigmas = torch.tensor(sigmas).float().to(self.device)
        s = torch.flip(sigmas, dims=(-1,))
        sigma = s[t]
        
        # total out
        total_out = kwargs.get('total_out')
        
        # 1. degradation
        num_sampling = 1
        norm = 0
        for _ in range(num_sampling):
            # noisy input
            x_0_hat = x_0_hat 

            # x 0 hat -> norm
            A_x = self.degradation(x_0_hat, self.n_src)
            
            # difference for gradient
            sig_diff = measurement - A_x
            norm_total = (torch.linalg.norm(sig_diff)) # l2
            # group norm
            seg_num =  512 # 256, 512
            norm_sig_group = self.segmented_l2_norm(sig_diff, num_segments=seg_num) * 0.05 
            
            # fft norm
            fft_diff_1 = self.difference_fft(x_0_hat, measurement, self.n_src, 
                                             n_fft = 1024, hop_length = 512, win_length = 1024)
            norm_fft_1 = (torch.linalg.norm(fft_diff_1)) * 0.1  # abs: * 0.1

            # norm
            norm_fft = norm_fft_1
            
            # all
            norm = norm_sig_group + norm_total + norm_fft
        ########################update
        # obtain grad update
        grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        with torch.no_grad():
            step_norm = grad.pow(2).mean().sqrt()   
    
        posterior_variance = self.sde.posterior_variance
        
        #----------------------------------------------------------
        # 2. grad separate norm, dps
        ##################### dps #####################
        epsilon = 1e-8
        # grad src
        grad_chunk = torch.split(grad, grad.size(-1) // self.n_src, dim=-1)
        # x_t srsc
        x_t_chunk = x_t.split(x_t.size(-1) // self.n_src, dim=-1)
        # x_0_hat src
        x_0_hat_chunk = x_0_hat.split(x_0_hat.size(-1) // self.n_src, dim=-1)
        updated_x_sources = []
        all_s_i = []
        for i in range(self.n_src):
            grad_i = grad_chunk[i]
            x_t_i = x_t_chunk[i]
            # norm sub grad
            norm_grad_i = torch.linalg.norm(grad_i)
            dir_grad_i = grad_i / (norm_grad_i + epsilon) # norm
            dir_grad_i = grad_i 
            s_i = 1.5 
            # update n src
            x_t_i = x_t_i - dir_grad_i * s_i
            updated_x_sources.append(x_t_i)
            all_s_i.append(s_i)
        x_t = torch.cat(updated_x_sources, dim=-1)
    

        return x_t, norm


    def segmented_l2_norm(self, signal, num_segments=4):
        segments = torch.chunk(signal, num_segments, dim=-1)
        seg_norms = [torch.linalg.norm(seg) for seg in segments]
        return sum(seg_norms)


    def difference_fft(self, x, y, n_src, n_fft, hop_length, win_length):
        """
        x, y: signal
        """
        min_sample_length = x.shape[-1] // n_src
        segments = torch.split(x, min_sample_length, dim=-1)
        degraded = sum(segments)

        n_fft = 512
        hop_length = 256
        win_length = 512
        window = torch.hann_window(win_length, device=x.device)

        stft_degraded = torch.stft(degraded.squeeze(1), n_fft=n_fft, hop_length=hop_length, 
                                    win_length=win_length, window=window, return_complex=True)
        stft_y = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=hop_length, 
                            win_length=win_length, window=window, return_complex=True)
        
        spec_degraded = stft_degraded.abs()
        spec_y = stft_y.abs()
        
        diff = (spec_degraded - spec_y).abs()  
        
        return diff


    def update_fn_recons_combin(self, n_src, x_t, x_prev, x_0_hat, measurement, **kwargs):
        # direct sample
        steps = 2
        total_log_sum = 0
        for i in range(steps):
            new_samples = []
            segments = torch.split(x_prev, measurement.size(-1), dim=-1) 
            x_sum = sum(segments)
            # log p(y | x)
            log_p_y_x = measurement - x_sum
            coefficient = 1/2
            total_log_sum += log_p_y_x
            start = 0
            end = measurement.size(-1)

            while end <= x_prev.size(-1):
                new_sample = (
                    x_t[:, :, start:end] + coefficient * total_log_sum
                )
                new_samples.append(new_sample)
                start = end
                end += measurement.size(-1)
            x_prev = torch.cat(new_samples, dim=-1)

        # degradation guidance
        A_x0 = self.degradation(x_0_hat, self.n_src)

        # signal time domain difference
        difference = measurement - A_x0
        sig_norm = torch.linalg.norm(difference)

        # frequency domain
        # fft_norm = self.difference_fft(x_0_hat, measurement, self.n_src)

        # model guidance
        t = kwargs.get('time')

        # norm
        norm = sig_norm

        # grad
        # norm_grad = torch.autograd.grad(outputs=spk_emb_value, inputs=x_concat)[0]
        
        # grad
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_0_hat)[0]

        # scale
        t = kwargs.get('time')
        sigma = torch.sqrt(self.alphas[t])
        
        self.scale = 1.0 # 1.0
        s = self.scale / (sigma+ 1e-6)
        # s = self.scale

        # normguide =  (x_0_hat.size(-1) ** 0.5)
        # normguide = torch.linalg.norm(norm_grad) / (x_0_hat.size(-1) ** 0.5)
        # s = self.xi / (normguide + 1e-6)
        # 
        # sigma = torch.sqrt(self.alphas[t])
        # s = self.xi / (normguide * sigma + 1e-6)

        # update
        x_t -= norm_grad * s
        
        return x_t, norm


    def update_spk_embed(self, n_src, x_t, x_prev, x_0_hat, measurement, **kwargs):
        # x 0 segments for different srcs
        x_prev_segments = torch.split(x_prev, x_0_hat.size(-1) // n_src, dim=-1) 
        torch.backends.cudnn.enabled = False
        x_concat = torch.cat(x_prev_segments, dim=1).squeeze(0) # [B, wave length]
        spk_emb_value = self.spk_model(x_concat)
        spk_emb_value = torch.sigmoid(spk_emb_value).mean()

        # grad
        norm_grad = torch.autograd.grad(outputs=spk_emb_value, inputs=x_concat)[0]
        torch.backends.cudnn.enabled = True

        norm_grad = norm_grad.reshape(1, 1, -1)

        # scale
        self.scale = 1.0 # 1.0
        t = kwargs.get('time')
        sigma = torch.sqrt(self.alphas[t])
        s = self.scale / (sigma + 1e-6)
        # update
        x_t -= norm_grad * s

        return x_t, spk_emb_value


    def update_fn_adaptive(
        self, n_src, x, x_prev, t, y, threshold=150, steps=1, model_kwargs_in=None, source_separation=False 
    ):
        x, condition = self.update_fn(n_src, x, x_prev, t, y, steps, source_separation)

        if t[0] < threshold and t[0] > 0:
            if self.sde.input_sigma_t:
                eps = self.score_fn(
                    x, _extract_into_tensor(self.sde.betas, t, t.shape)
                )
            else:
                model_kwargs = model_kwargs_in
                x = x.squeeze(1)
                
                # to batch
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) # [single src,...]
                x_stack = torch.cat(x_solo, dim=0)
                t_b = torch.tensor([t] * x_stack.size(0), device=x_stack.device) 
                eps = self.score_fn(x_stack, self.sde._scale_timesteps(t_b, True), **model_kwargs)
                if isinstance(eps, tuple):
                    eps = eps[0]
                    eps = eps.reshape(1, 1, -1)
                else:
                    eps = eps.reshape(1, 1, -1)
                
                x = x.unsqueeze(1)
                
            if condition is None:
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)

                condition = y - (x_sum) 

            if source_separation:
                x = self.langevin_corrector_sliced(x, t, eps, y, condition)
            else:
                x = self.langevin_corrector(x, t, eps, y, condition)

        return x


    def update_fn(self, n_src, x, x_prev, t, y, steps, source_separation):
        if source_separation:
            coefficient = 0.1 # 0.5
            total_log_sum = 0
            # x: out (x(i-1)) sampling, x_prev: previous x
            for i in range(steps):
                new_samples = []
                
                # noise
                # sigma = torch.sqrt(self.alphas[t]).float()
                # x_prev = x_prev + 0.05 * sigma * torch.randn_like(x_prev)

                segments = torch.split(x_prev, y.size(-1), dim=-1) 
                x_sum = sum(segments)

                # log p(y | x)
                log_p_y_x = y - x_sum
                total_log_sum += log_p_y_x

                start = 0
                end = y.size(-1)
                while end <= x_prev.size(-1):
                    new_sample = (
                        x["sample"][:, :, start:end] + coefficient * total_log_sum
                    )
                    new_samples.append(new_sample)
                    start = end
                    end += y.size(-1)
                x_prev = torch.cat(new_samples, dim=-1)
            condition = None

        return x_prev.float(), condition


    def update_fn_coeff(self, n_src, x, x_prev, t, y, steps, source_separation):
        if source_separation:
            coefficient = 0.5 # 0.5
            total_log_sum = 0
            # x: out (x(i-1)) sampling, x_prev: previous x
            for i in range(steps):
                new_samples = []
                
                segments = torch.split(x_prev, y.size(-1), dim=-1) 
                x_sum = sum(segments)
                x_sum = x_sum * _extract_into_tensor(self.sde.sqrt_recip_alphas_cumprod, t, x_sum.shape)

                # log p(y | x)
                log_p_y_x = y - x_sum
                log_p_y_x = log_p_y_x * _extract_into_tensor(self.sde.sqrt_alphas_cumprod, t, x_sum.shape)

                # coefficient = 1/ (2 * (1 - self.sde.alphas_cumprod[t]))
                coefficient = 1/2

                total_log_sum += log_p_y_x

                start = 0
                end = y.size(-1)
                while end <= x_prev.size(-1):
                    new_sample = (
                        x["sample"][:, :, start:end] + coefficient * total_log_sum
                    )
                    new_samples.append(new_sample)
                    start = end
                    end += y.size(-1)
                x_prev = torch.cat(new_samples, dim=-1)
            condition = None

        return x_prev.float(), condition



    def langevin_corrector_sliced(self, x, t, eps, y, condition=None):
        alpha = self.alphas[t]
        corrected_samples = []

        start = 0
        end = y.size(-1)
        while end <= x.size(-1):
            score = self.recip_alphas[t] * eps[:, :, start:end]
            noise = torch.randn_like(x[:, :, start:end], device=x.device)
            grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            score_to_use = score + condition if condition is not None else score 
            x_new = (
                x[:, :, start:end]
                + step_size * score_to_use
                + torch.sqrt(2 * step_size) * noise
            )
            corrected_samples.append(x_new)
            start = end
            end += y.size(-1)

        return torch.cat(corrected_samples, dim=-1).float()


    def langevin_corrector(self, x, t, eps, y, condition=None):
        alpha = self.alphas[t]

        score = self.recip_alphas[t] * eps
        noise = torch.randn_like(x, device=x.device)
        grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha

        score_to_use = score + condition if condition is not None else score
        x_new = x + step_size * score_to_use + torch.sqrt(2 * step_size) * noise

        return x_new.float()


