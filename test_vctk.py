import os
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.set_device(0)
print("VISIBLE:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_VISIBLE inside container:", torch.cuda.device_count())
import torchaudio
import toml
import json
import random
import utils
import diffusion
import importlib
import numpy as np
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from loss_metric import _calculate_sisnr, pit_sisnr
# models
import models
# seeds
seed = 7
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  


def _get_VCTK_paths(root_dir, used_spk_num, test = True):
    spk_paths = []
    speech_paths = []
    i = 0
    for spk in sorted(os.listdir(root_dir)):
        spk_dir = os.path.join(root_dir, spk)
        if os.path.isdir(spk_dir):
            spk_utterances = []
            for f in os.listdir(spk_dir): # walk all
                if f.endswith('.flac') or f.endswith('.wav'):
                    audio_path = os.path.join(spk_dir, f)
                    spk_utterances.append(audio_path)
                    speech_paths.append(audio_path)
            spk_paths.append(spk_utterances)
        i += 1
    
    len_data = len(spk_paths) # spk num, 110
    random.shuffle(spk_paths)
    
    # this is for vctk test dataset
    # if test:
    #     spk_paths = spk_paths[used_spk_num:]
    # else:
    #     spk_paths = spk_paths[:used_spk_num] 
    
    print('num spk', len(spk_paths))
    print('num speech', len(speech_paths))
    return spk_paths, speech_paths


def make_data(
    speech_data,
    n_src,
    file_sr,
    tgt_sr,
    tgt_len_s
):
    target_frames = int(tgt_len_s * tgt_sr)

    loaded_audios = []
    total_label = []

    choosen_speaker = random.sample(speech_data, n_src)  # each item: (path, spk_id)
    choosen_speeches = [random.choice(one_speaker) for one_speaker in choosen_speaker]

    for speech_path in choosen_speeches:
        audio, sr = torchaudio.load(speech_path)
        if sr != tgt_sr:
            audio = torchaudio.transforms.Resample(sr, tgt_sr)(audio)

        if audio.shape[-1] > target_frames:
            start = random.randint(0, audio.shape[-1] - target_frames)
            audio = audio[..., start : start + target_frames]
        else:
            pad_len = target_frames - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad_len))

        loaded_audios.append(audio)

    total_src_on_canvas = []
    last_placement_info = {'start': -1, 'end': -1}

    for i, audio in enumerate(loaded_audios):
        current_frames = audio.shape[-1]
        canvas = torch.zeros(1, target_frames, device=audio.device)

        if i == 0:
            start_frame = random.randint(0, target_frames - current_frames)
        else:
            prev_start = last_placement_info['start']
            prev_end = last_placement_info['end']
            
            # overlap range
            min_start = max(0, prev_start)
            max_start = min(prev_end, target_frames - current_frames)

            if min_start >= max_start:
                start_frame = min(target_frames - current_frames, prev_start)
            else:
                start_frame = random.randint(min_start, max_start)

        canvas[..., start_frame : start_frame + current_frames] = audio
        last_placement_info['start'] = start_frame
        last_placement_info['end'] = start_frame + current_frames

        canvas, _ = change_energy(canvas)
        total_src_on_canvas.append(canvas.squeeze(0))  # [T]

    single_sources = torch.stack(total_src_on_canvas, dim=0)  # [n_src, T]
    mixture = single_sources.sum(dim=0)  # [T]

    return mixture, single_sources, total_label


def change_energy(audio):
    current_rms = torch.sqrt(torch.mean(audio**2))
    current_rms_db = 20 * torch.log10(current_rms)
    
    # tgt db
    tgt_db = round(random.uniform(-25, -20), 3)
    
    # db diff
    db_difference = tgt_db - current_rms_db 
    amplify_ratio = 10 ** (db_difference / 20)
    audio = audio * amplify_ratio
    
    # info
    info = {'amplify_ratio':amplify_ratio, 'tgt_db':tgt_db}
    
    return audio, info


def degradation(x: torch.Tensor, n_src = 2) -> torch.Tensor:
    min_sample_length = x.shape[-1] // n_src
    segments = torch.split(x, min_sample_length, dim=-1)  # [..., min_sample_length]
    degraded = sum(segments)
    return degraded


def _norm(audio, eps: float = 1e-8):
    peak = audio.abs().amax(dim=-1, keepdim=True)
    peak = peak.clamp(min=eps)
    audio = audio / peak
    audio = audio * 0.95
    return audio, 0.95 / peak


def evaluate(model, 
             diffusion, 
             device, 
             speech_data, 
             
             save_root):
    file_sr = 44100 # vctk Hz
    tgt_sr = 16000 # Hz
    total_snr = []
    mean_snr = []
    all_mean_snr = []
    
    # save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_root, timestamp)
    os.makedirs(save_path, exist_ok=True)
    
    # sample num
    sample_num = 2 
    
    all_snr = []
    for item in tqdm(range(sample_num), total=sample_num):
        # make data, mixture
        file_sr = 44100 # Hz
        tgt_sr = 16000 # Hz
        tgt_len_sec = 4 # 4 s
        n_src = 2  # 2
        mixture, single_sources, total_label = make_data(speech_data=speech_data,
                                                                        n_src=n_src,
                                                                        file_sr=44100,
                                                                        tgt_sr=16000,
                                                                        tgt_len_s=tgt_len_sec)
        
        # norm
        mixture, peak = _norm(mixture)
        mixture = mixture.to(device)
        mixture = mixture.reshape(1, 1, -1)

        # model label
        n_src_num = n_src 
        model_kwargs_in = None
        
        # pspeaking embedding guidance
        generator = diffusion.dirac_sample_dps_eval(
                                            model,
                                            shape = mixture.shape,
                                            measurement = mixture,
                                            n_src = n_src_num,
                                            degradation = degradation,
                                            model_kwargs = model_kwargs_in)


        final_sample = None
        for out in generator:  
            final_sample = out["sample"]
        sepa_srcs = final_sample.cpu() 
        sepa_srcs_chunked = torch.chunk(
            sepa_srcs, chunks = n_src_num, dim=-1
        )

        # metric
        estimated_sources = torch.stack([sepa_srcs_chunked[i].squeeze(0) for i in range(n_src_num)])
        estimated_sources = estimated_sources.squeeze(1)
        reference_sources = torch.stack([(single_sources[i] * peak).squeeze(0) for i in range(n_src_num)])

        # pit SI-SNR
        avg_sisnr, best_perm = pit_sisnr(estimated_sources, reference_sources)
        
        # save wav
        single_mean_snr = 0
        for i in range(n_src_num):
            best_idx = best_perm[i]
            estimated_source = estimated_sources[best_idx]
            reference_source = reference_sources[i]
            
            sisnr_value = _calculate_sisnr(estimated_source, reference_source)
            sisnr_value = float(sisnr_value)
            total_snr.append(sisnr_value)
            single_mean_snr += sisnr_value
            
            # save audio
            save_path_root = os.path.join(save_path, str(item))
            os.makedirs(save_path_root, exist_ok=True)
            save_wave_path = os.path.join(save_path_root, f"sepa_{i}.wav")
            torchaudio.save(save_wave_path, estimated_source.unsqueeze(0).to('cpu'), tgt_sr)
            save_wave_path = os.path.join(save_path_root, f"real_{i}.wav")
            torchaudio.save(save_wave_path, reference_source.unsqueeze(0).to('cpu'), tgt_sr)
            
        save_wave_path = os.path.join(save_path_root, f"mix.wav")
        mixture_save = mixture.squeeze(0).cpu() 
        torchaudio.save(save_wave_path, mixture_save, tgt_sr) # mixture
        mean_snr.append(single_mean_snr / n_src_num)
        all_snr.append(single_mean_snr / n_src_num)
            
        # --- Save all SI-SNR lists to a JSON file ---
        all_snr_mean = sum(all_snr) / len(all_snr)
        snr_filename = os.path.join(save_path, "si_snr_results.json")
        current_result = {
            'all_snr_mean': all_snr_mean,
            "total_snr": total_snr,
            "mean_snr": mean_snr
        }
        with open(snr_filename, 'w') as f:
            json.dump(current_result, f, indent=4) 

    print(f"Saved SI-SNR results to {snr_filename}")
            
            
def load_model(config, device):
    model_name = config["model_name"]
    model_class = getattr(models, model_name)
    model = model_class(config["model_cfg"])
    for param in model.parameters():
        param.requires_grad = False

    # paras info
    print('model:',model_name, 'paras num:', utils.get_paras_num(model))
    print('ckpt path:', config["ckpt_path"])
    
    # load state_dict
    ckpt = torch.load(config["ckpt_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # ema
    ema_model = deepcopy(model)
    ema_model.load_state_dict(ckpt["ema"])
    ema_model.to(device).eval()
    
    return ema_model


def load_one_model(config_path, device):
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f) if ext == '.json' else toml.load(f)
    return load_model(config, device)


def main(device,
         speech_dir,
         speech_config_path,
         save_root):
        
    # speech list
    speech_dir = speech_dir
    spk_paths, speech_paths = _get_VCTK_paths(speech_dir, used_spk_num = 100)
    speech_data = [] # wav_path, spk_id = speech_data[idx]
    for spk_id, utterances in enumerate(spk_paths):
        for u in utterances:
            speech_data.append((u, spk_id))

    # load models
    # vctk model
    speech_config_path = speech_config_path
    speech_model = load_one_model(speech_config_path, device)
    model = speech_model

    # diffusion
    config_path = speech_config_path
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f) if ext == '.json' else toml.load(f)
    gaus_diffusion = diffusion.GaussianDiffusion_spk(
                                steps = 200, 
                                config_file = config,
                                beta_start = config['train_para']['beta_start'],
                                beta_end = config['train_para']['beta_end'],
                                device = device)
    
    # target output folder
    save_root = save_root
    
    evaluate(model, gaus_diffusion, device, 
             speech_data = spk_paths,
             save_root = save_root)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    speech_dir = './speech_samples'
    speech_config_path = './configs/config.toml'
    save_root = './test_results'
    
    main(device,
         speech_dir,
         speech_config_path,
         save_root)

    