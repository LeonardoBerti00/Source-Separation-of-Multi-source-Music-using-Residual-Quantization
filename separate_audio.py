#!/usr/bin/env python3

import os
import sys
import torch
import torchaudio
import logging
import librosa
import numpy as np
import argparse
from pathlib import Path
# Removed unused import

# Project imports
import constants as cst
from utils.utils_transformer import load_autoencoder
# Removed unused import


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resample_if_needed(audio_array, src_sr, target_sr=22050):
    """Resample audio if the sample rate doesn't match the target"""
    if src_sr != target_sr:
        logging.info(f"Resampling audio from {src_sr}Hz to {target_sr}Hz")
        return librosa.resample(audio_array, orig_sr=src_sr, target_sr=target_sr)
    return audio_array

def load_audio_file(file_path, target_sr=22050):
    """Load audio file and ensure it's correctly formatted"""
    logging.info(f"Loading audio file: {file_path}")
    
    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to numpy for processing
    audio_np = waveform.numpy()
    
    # Resample if needed
    if sample_rate != target_sr:
        channels = []
        for channel in audio_np:
            channels.append(resample_if_needed(channel, sample_rate, target_sr))
        audio_np = np.stack(channels)
    
    # Convert to mono if needed by averaging channels
    if audio_np.shape[0] > 1:
        logging.info(f"Converting {audio_np.shape[0]} channels to mono by averaging")
        audio_np = np.mean(audio_np, axis=0, keepdims=True)
    elif audio_np.shape[0] < 1:
        raise ValueError("Audio file has no channels")
        
    # Convert back to torch
    audio_tensor = torch.from_numpy(audio_np).float()
    
    return audio_tensor, target_sr

def separate_audio(model, mixture, device, z_score=True):
    """Process audio through the model to separate sources"""
    logging.info("Running source separation with RQVAE model")
    
    # Ensure exact sample length
    mixture_len = mixture.shape[-1]
    if mixture_len != cst.SAMPLE_LENGTH:
        logging.info(f"Adjusting input length from {mixture_len} to exactly {cst.SAMPLE_LENGTH} samples")
        # Pad or trim to exact sample length
        if mixture_len < cst.SAMPLE_LENGTH:
            # Pad
            pad_amount = cst.SAMPLE_LENGTH - mixture_len
            mixture = torch.nn.functional.pad(mixture, (0, pad_amount))
        else:
            # Trim
            mixture = mixture[..., :cst.SAMPLE_LENGTH]
    
    # Make batch dimension
    if mixture.ndim == 2:  # [channels, samples]
        mixture = mixture.unsqueeze(0)  # [batch, channels, samples]
    
    # Move to device
    mixture = mixture.to(device)
    
    # Apply z-score normalization if needed
    if z_score:
        mixture = (mixture - cst.MEAN) / cst.STD
    
    # Forward pass through model
    with torch.no_grad():
        # We need to ensure correct shape for model input
        recon, _ = model.forward(mixture, is_train=False, batch_idx=0)
    
    # Return the four stems (batch dimension removed)
    stems = [recon[0, i] for i in range(len(cst.STEMS))]
    
    # Denormalize if needed
    if z_score:
        stems = [(stem * cst.STD) + cst.MEAN for stem in stems]
    
    return stems

def process_in_chunks(model, audio_tensor, device='cuda'):
    """Process long audio files in chunks using exact SAMPLE_LENGTH sized chunks"""
    logging.info(f"Processing audio in chunks of {cst.SAMPLE_LENGTH} samples")
    
    # Get audio length and ensure it's at least mono
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    _, audio_length = audio_tensor.shape
    
    # Use fixed chunk size to match model's expected input length
    chunk_size = cst.SAMPLE_LENGTH
    overlap = chunk_size // 4  # 25% overlap for smooth transitions
    
    # Initialize output tensors for each stem
    stem_outputs = [torch.zeros(audio_length) for _ in range(len(cst.STEMS))]
    
    # Initialize normalization tensor to handle overlaps
    norm_tensor = torch.zeros(audio_length)
    
    # Define window function for smooth blending at overlapping regions
    fade_size = overlap * 2
    window = torch.hann_window(fade_size)
    
    # Process chunks
    hop_size = chunk_size - overlap
    for start_pos in range(0, audio_length, hop_size):
        # Define chunk boundaries
        end_pos = min(start_pos + chunk_size, audio_length)
        
        # If we're at the end with a small chunk, break
        # We'll handle the remainder separately
        if end_pos - start_pos < chunk_size // 2:
            break
            
        # Extract chunk
        chunk = audio_tensor[:, start_pos:end_pos]
        
        # Process chunk (separate_audio will handle padding to exact size)
        stems = separate_audio(model, chunk, device)
        
        # Get the actual size of processed audio (could be chunk_size)
        actual_len = min(chunk.shape[1], stems[0].shape[0])
        
        # Apply windows for smooth transitions
        for i, stem in enumerate(stems):
            stem_chunk = stem.cpu()
            
            # Only use the part that corresponds to our input
            stem_chunk = stem_chunk[:actual_len]
            
            # Apply fade in/out at chunk boundaries
            if start_pos > 0:  # Apply fade-in if not the first chunk
                fade_in = min(fade_size, stem_chunk.shape[0])
                stem_chunk[:fade_in] *= window[:fade_in]
                
            if end_pos < audio_length:  # Apply fade-out if not the last chunk
                fade_out = min(fade_size, stem_chunk.shape[0])
                stem_chunk[-fade_out:] *= window[-fade_out:]
            
            # Add to output
            end_idx = min(start_pos + stem_chunk.shape[0], audio_length)
            stem_outputs[i][start_pos:end_idx] += stem_chunk[:end_idx-start_pos]
            
            # Update normalization tensor for this region
            norm_chunk = torch.ones_like(stem_chunk[:end_idx-start_pos])
            if start_pos > 0:  # Fade-in normalization
                fade_in = min(fade_size, norm_chunk.shape[0])
                norm_chunk[:fade_in] *= window[:fade_in]
                
            if end_pos < audio_length:  # Fade-out normalization
                fade_out = min(fade_size, norm_chunk.shape[0])
                norm_chunk[-fade_out:] *= window[-fade_out:]
                
            norm_tensor[start_pos:end_idx] += norm_chunk
    
    # Handle the remainder (last partial chunk)
    if audio_length % hop_size != 0:
        start_pos = max(0, audio_length - chunk_size)
        chunk = audio_tensor[:, start_pos:audio_length]
        stems = separate_audio(model, chunk, device)
        
        # Apply fade-in only for the overlapping part
        for i, stem in enumerate(stems):
            stem_chunk = stem.cpu()[:audio_length-start_pos]
            
            # Apply fade-in for the overlapping part
            fade_in = min(fade_size, stem_chunk.shape[0])
            stem_chunk[:fade_in] *= window[:fade_in]
            
            # Add to output
            stem_outputs[i][start_pos:audio_length] += stem_chunk
            
            # Update normalization tensor
            norm_chunk = torch.ones_like(stem_chunk)
            norm_chunk[:fade_in] *= window[:fade_in]
            norm_tensor[start_pos:audio_length] += norm_chunk
    
    # Normalize to avoid overlap issues
    norm_tensor[norm_tensor < 0.001] = 1.0  # Avoid division by zero
    for i in range(len(stem_outputs)):
        stem_outputs[i] = stem_outputs[i] / norm_tensor
    
    return stem_outputs

def main():
    parser = argparse.ArgumentParser(description='Separate instruments in an audio file using RQVAE model')
    parser.add_argument('audio_file', type=str, help='Path to audio file to separate')
    parser.add_argument('--output_dir', type=str, default='data/separations',
                        help='Directory to save separated audio files')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()

    # Check if file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logging.error(f"Audio file not found: {audio_path}")
        return 1
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = 'cpu' if args.use_cpu or not torch.cuda.is_available() else 'cuda'
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info("Loading RQVAE model...")
    try:
        model, config = load_autoencoder(cst.Autoencoders.RQVAE.name)
        model.eval()
        model.to(device)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return 1

    # Load audio
    try:
        audio_tensor, sr = load_audio_file(audio_path)
        logging.info(f"Audio loaded: {audio_tensor.shape}, {sr}Hz")
    except Exception as e:
        logging.error(f"Failed to load audio: {e}")
        return 1
    
    # Process audio
    try:
        stems = process_in_chunks(model, audio_tensor, device=device)
        
        # Save separated stems
        logging.info(f"Saving separated stems to {args.output_dir}")
        for stem_name, stem_audio in zip(cst.STEMS, stems):
            output_file = f"{args.output_dir}/separated_{audio_path.stem}_{stem_name}.wav"
            torchaudio.save(output_file, stem_audio.unsqueeze(0), sr)
            logging.info(f"Saved {output_file}")
            
        logging.info("Separation complete!")
        return 0
        
    except Exception as e:
        logging.error(f"Error during separation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
