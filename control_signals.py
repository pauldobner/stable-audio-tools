import librosa
import numpy as np
import torch
import torchcrepe


def compute_loudness(audio: np.ndarray, sr: int, hop_length: int = None, n_fft: int = None, win_length: int = None) -> np.ndarray:
    """
    Compute per-frame loudness as the RMS of an A-weighted magnitude spectrogram.
    Uses A-weighting to mimic human hearing.
    Inspired by the torchcrepe implementation, but uses RMS instead of the mean.
    https://github.com/maxrmorrison/torchcrepe/blob/master/torchcrepe/loudness.py
    """
    if hop_length is None:
        hop_length = int(sr / 200.)  # 5ms per frame
    if n_fft is None:
        n_fft = 1024  # default FFT size
    if win_length is None:
        win_length = n_fft  # default window length

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Compute A-weighting (in dB)
    A_weight_db = librosa.A_weighting(freqs)

    # convert magnitude to db
    S_db = librosa.amplitude_to_db(S)

    # applies A-weighting to each frequency bin in the spectrogram
    S_weighted = S_db + A_weight_db[:, np.newaxis]
    # Compute RMS across frequency bins
    loudness = np.sqrt(np.mean(S_weighted ** 2, axis=0))
    return loudness


def spectral_centroid(y: np.ndarray, sr: int, hop_length: int = None, n_fft: int = None, win_length: int = None, midi: bool = True) -> np.ndarray:
    """
    Compute the spectral centroid per frame and convert it to MIDI values (semitones),
    scaled to roughly (0, 1) by dividing by 127.

    This center of mass directly correlates with perceived brightness or timbre:
    High spectral centroid: Energy concentrated in higher frequencies = brighter/sharper sound
    Low spectral centroid: Energy concentrated in lower frequencies = darker/mellower sound

    """
    if hop_length is None:
        hop_length = int(sr / 200.)  # 5ms per frame
    if n_fft is None:
        n_fft = 1024  # default FFT size
    if win_length is None:
        win_length = n_fft  # default window length
        
    centroid_hz = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, win_length=win_length)[0]
    # Ensure no zero values for safe logarithm calculation
    centroid_hz = np.maximum(centroid_hz, 1e-6)
    if midi:
        centroid_midi = 69 + 12 * np.log2(centroid_hz / 440.0)
        # scale to roughly (0, 1) by dividing by 127
        centroid_midi = centroid_midi / 127.0
        return centroid_midi
    return centroid_hz


def extract_pitch_probability(audio: np.ndarray | torch.Tensor, sr: int, hop_length: int = None, model: str = 'tiny', device: str = 'cuda:0') -> np.ndarray:
    """
    Extract the raw pitch probabilities (and hence periodicity) using torchcrepe.

    Uses the CREPE "tiny" variant. Zeroes out probabilities below 0.1 per the paper.
    """
    # Ensure audio is a torch tensor
    if not torch.is_tensor(audio):
        audio = torch.tensor(audio, dtype=torch.float32)
    # make sure shape is (1, n_samples)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim > 2:
        raise ValueError("Audio tensor must be 1D or 2D")

    # Use a 5 ms hop (i.e., sr/200) if not specified
    if hop_length is None:
        hop_length = int(sr / 200.)

    # torchcrepe.preprocess returns a generator of batches; get the first batch.
    batch = next(torchcrepe.preprocess(audio, sr, hop_length))
    batch = batch.to(device)
    with torch.no_grad():
        # Call torchcrepe.infer with proper parameters as per the repository
        pitch_probs = torchcrepe.infer(batch, model=model, device=device)

    # Zero out all probabilities below 0.1 to avoid leaking timbral information
    pitch_probs = torch.where(
        pitch_probs < 0.1, torch.zeros_like(pitch_probs), pitch_probs)
    pitch_probs = pitch_probs.cpu()
    torch.cuda.empty_cache()
    return pitch_probs.numpy()