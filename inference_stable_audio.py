import json
import logging
import os
import shutil
import time
from datetime import timedelta

import soundfile as sf
import torch
import torchaudio
from einops import rearrange
from tqdm import tqdm

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from utils.vimsketch_dataset import VimSketchDataset

# Constants
MODEL_CONFIG_PATH = "hf_model_download/model_config_small_original.json"
MODEL_CKPT_PATH = "hf_model_download/model_small.safetensors"
DATASET_ROOT = "/home/paul/OneDrive/Master/practical_work/Practical-Work-AI-Master/dataset/Vim_Sketch_Dataset/"

# Inference Parameters
TRANSFER_STRENGTH = 0.75 # for small model from 0 to 1
GUIDANCE_SCALE = 1.0 # 1.0 for rf_denoiser
STEPS = 8 # 8 for rf_denoiser
SEED = 42
TTA = False # Set to True for Text-to-Audio, False for Style Transfer

def load_model(model_config_path, model_ckpt_path, device="cuda"):
    print(f"Loading model config from {model_config_path}")
    with open(model_config_path) as f:
        model_config = json.load(f)

    print(f"Creating model from config")
    model = create_model_from_config(model_config)

    print(f"Loading model checkpoint from {model_ckpt_path}")
    copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
    
    model.to(device).eval().requires_grad_(False)
    print("Model loaded successfully")
    return model, model_config

def save_wave(waveform, savepath, name="outwav", sample_rate=44100):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        path = os.path.join(
            savepath,
            "%s.wav"
            % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0]
            ),
        )
        print("Save audio to %s" % path)
        
        # Post-processing from inftest.py
        # waveform[i] is [channels, samples]
        audio = waveform[i]
        
        # Peak normalize, clip, convert to int16
        # from https://huggingface.co/stabilityai/stable-audio-open-small
        audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        torchaudio.save(path, audio, sample_rate)

if __name__ == "__main__":
    # Setup paths
    # Allow overriding dataset root via env var or just use default
    dataset_root = os.environ.get("DATASET_ROOT", DATASET_ROOT)
    if not os.path.exists(dataset_root):
        # Fallback to checking if the user provided path exists, otherwise warn
        print(f"Warning: Dataset root {dataset_root} does not exist. Please check configuration.")

    dataset = VimSketchDataset(dataset_root)
    
    if TTA:
        save_path = os.path.join(dataset_root, "tta_sao", "audios")
    else:
        save_path = os.path.join(dataset_root, "style_transfer_sao", f"transfer_strength_{TRANSFER_STRENGTH}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Setup logging
    if TTA:
        log_filename = "processing_tta.log"
        cache_filename = "tta_cache.json"
        checkpoint_filename = "checkpoint_tta.json"
    else:
        log_filename = f"processing_transfer_{TRANSFER_STRENGTH}.log"
        cache_filename = f"transfer_cache_{TRANSFER_STRENGTH}.json"
        checkpoint_filename = f"checkpoint_transfer_{TRANSFER_STRENGTH}.json"

    log_file = os.path.join(save_path, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Checkpoint file to track progress
    checkpoint_file = os.path.join(save_path, checkpoint_filename)
    # Cache file to store tta_cache between runs
    cache_file = os.path.join(save_path, cache_filename)
    start_idx = 0
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            start_idx = checkpoint_data.get('last_processed_idx', 0) + 1
            logging.info(f"Resuming from index {start_idx} ({start_idx}/{len(dataset)} files)")
    
    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model on {device}")
    model, model_config = load_model(MODEL_CONFIG_PATH, MODEL_CKPT_PATH, device=device)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    # Start timing
    start_time = time.time()
    
    try:
        # Initialize the cache and load from file if exists
        tta_cache = {}
        
        # load cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Use a more robust format: store text and duration as separate keys
                    for entry in cache_data:
                        text = entry["text"]
                        duration = entry["duration"]
                        file_path = entry["file_path"]
                        
                        # Validate that the cached file still exists
                        if os.path.exists(file_path):
                            tta_cache[(text, duration)] = file_path
                        
                    logging.info(f"Loaded {len(tta_cache)} valid entries from cache file")
            except Exception as e:
                logging.error(f"Error loading cache file: {str(e)}")
                tta_cache = {}
        
        # Function to save cache to file
        def save_cache(force=False):
            try:
                # Convert to a more robust format
                cache_data = []
                for (text, duration), file_path in tta_cache.items():
                    cache_data.append({
                        "text": text,
                        "duration": duration,
                        "file_path": file_path
                    })
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logging.debug(f"Cache saved with {len(tta_cache)} entries")
            except Exception as e:
                logging.error(f"Error saving cache file: {str(e)}")

        for i in tqdm(range(start_idx, len(dataset))):
            imitation_path = dataset[i]["imitation_path"]
            reference_path = dataset[i]["reference_path"]
            text = dataset[i]["text"]
            
            # Get output filename
            imitation_filename = os.path.basename(imitation_path)
            output_file = os.path.join(save_path, imitation_filename)
            
            # Skip if already processed
            if os.path.exists(output_file):
                logging.info(f"Skipping {imitation_filename}, already processed")
                continue

            try:
                # Get audio duration from imitation path
                # We can use torchaudio or soundfile to get duration
                info = torchaudio.info(imitation_path)
                duration = info.num_frames / info.sample_rate
                
                # Prepare conditioning
                conditioning = [{"prompt": text, "seconds_start": 0, "seconds_total": duration}]
                
                # Calculate sample_size based on duration
                # Stable Audio Open generates fixed size chunks usually, but we can try to match duration
                # However, the model has a max sample size. 
                # Let's stick to the model's sample_size or adjust if needed.
                # For now, we will use the duration to calculate target sample size, but capped at model's max if needed?
                # Actually, generate_diffusion_cond takes sample_size.
                target_sample_size = int(duration * sample_rate)
                
                # Ensure target_sample_size is valid for the model (e.g. divisible by downsampling ratio)
                if model.pretransform is not None:
                    downsampling_ratio = model.pretransform.downsampling_ratio
                    # Make sure it's a multiple of downsampling ratio
                    target_sample_size = (target_sample_size // downsampling_ratio) * downsampling_ratio

                # Determine sampler parameters based on model objective
                diffusion_objective = model.diffusion_objective
                if diffusion_objective == "rf_denoiser":
                    sampler_type = "pingpong"
                    sigma_min = 0.01 # Not used for RF but good to have defined
                    sigma_max = 1.0
                elif diffusion_objective == "rectified_flow":
                    sampler_type = "euler"
                    sigma_min = 0.01
                    sigma_max = 1.0
                else:
                    sampler_type = "dpmpp-3m-sde"
                    sigma_min = 0.03
                    sigma_max = 500

                if TTA:
                    # Stable Audio Open logic
                    # Calculate max duration from model config
                    max_duration = sample_size / sample_rate
                    
                    if duration > max_duration:
                        logging.warning(f"Requested duration {duration:.2f}s exceeds model max {max_duration:.2f}s, clipping.")
                        duration = max_duration
                        
                    target_sample_size = int(duration * sample_rate)
                    if model.pretransform is not None:
                        downsampling_ratio = model.pretransform.downsampling_ratio
                        target_sample_size = (target_sample_size // downsampling_ratio) * downsampling_ratio

                    # if the audio with this text and duration already exists, copy it based on the filename
                    if (text, duration) in tta_cache:
                        outfile_path_cache = tta_cache[(text, duration)]
                        if os.path.exists(outfile_path_cache):
                            logging.info(f"Copying {imitation_filename}, was already processed")
                            shutil.copy(outfile_path_cache, output_file)
                            # Update checkpoint
                            with open(checkpoint_file, 'w') as f:
                                json.dump({'last_processed_idx': i, 'timestamp': time.time()}, f)
                            continue

                    # Generate TTA
                    output = generate_diffusion_cond(
                        model,
                        steps=STEPS,
                        cfg_scale=GUIDANCE_SCALE,
                        conditioning=conditioning,
                        sample_size=target_sample_size,
                        seed=SEED,
                        device=device,
                        init_audio=None,
                        init_noise_level=1.0,
                        sampler_type=sampler_type,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                    )
                    
                    # Trimming
                    output = output[..., :target_sample_size]
                    
                    save_wave(output, save_path, name=imitation_filename, sample_rate=sample_rate)
                    # Add to cache and save
                    tta_cache[(text, duration)] = output_file
                    save_cache()
                
                else:
                    # Style transfer
                    # Load init audio
                    init_audio_tensor, init_sr = torchaudio.load(imitation_path)
                    
                    output = generate_diffusion_cond(
                        model,
                        steps=STEPS,
                        cfg_scale=GUIDANCE_SCALE,
                        conditioning=conditioning,
                        sample_size=target_sample_size,
                        seed=SEED,
                        device=device,
                        init_audio=(init_sr, init_audio_tensor),
                        init_noise_level=TRANSFER_STRENGTH,
                        sampler_type=sampler_type,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                    )
                    
                    # Trimming
                    output = output[..., :target_sample_size]
                    
                    save_wave(output, save_path, name=imitation_filename, sample_rate=sample_rate)              
                
                # Update checkpoint
                with open(checkpoint_file, 'w') as f:
                    json.dump({'last_processed_idx': i, 'timestamp': time.time()}, f)
                
                # log created files
                logging.info(f"Created {save_path}/{imitation_filename}")
                
                # Periodic status report
                if (i - start_idx) % 5 == 0 or i == len(dataset) - 1:
                    elapsed = time.time() - start_time
                    processed = i - start_idx + 1
                    if processed > 0:
                        avg_time_per_file = elapsed / processed
                        remaining_files = len(dataset) - i - 1
                        est_remaining = avg_time_per_file * remaining_files
                        
                        elapsed_str = str(timedelta(seconds=int(elapsed)))
                        remaining_str = str(timedelta(seconds=int(est_remaining)))
                        
                        logging.info(f"Processed {processed} files. Progress: {i+1}/{len(dataset)}")
                        logging.info(f"Time elapsed: {elapsed_str}, Est. remaining: {remaining_str}")
            
            except Exception as e:
                logging.error(f"Error processing {imitation_filename}: {str(e)}")
                # raise e # Uncomment for debugging
                continue
                
    except KeyboardInterrupt:
        logging.info("\nProcessing interrupted. Progress saved. You can resume later.")
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        raise e
        
    finally:
        # Save cache on exit to avoid losing recent entries
        if 'save_cache' in locals():
            save_cache(force=True)
        total_time = time.time() - start_time
        logging.info(f"Process finished or interrupted. Total runtime: {str(timedelta(seconds=int(total_time)))}")
