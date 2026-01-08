import librosa
import librosa.display
import numpy as np
import torch
import torchcrepe
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter as scipy_median_filter


def apply_median_filter(signal: np.ndarray, win_length: int) -> np.ndarray:
    """
    Apply a median filter to the signal along the time dimension.
    Supports 1D (time,) and 3D (batch, time, features) arrays.
    """
    if win_length <= 1:
        return signal

    if signal.ndim == 1:
        # 1D array: (time,)
        return scipy_median_filter(signal, size=win_length, mode='nearest')
    elif signal.ndim == 3:
        # 3D array: (batch, time, features)
        # Filter along axis 1
        return scipy_median_filter(signal, size=(1, win_length, 1), mode='nearest')
    else:
        # Fallback for 2D (time, features)
        if signal.ndim == 2:
             return scipy_median_filter(signal, size=(win_length, 1), mode='nearest')
        raise ValueError(f"Unsupported signal shape for median filtering: {signal.shape}")


def compute_loudness(
    audio: np.ndarray,
    sr: int,
    hop_length: int = None,
    n_fft: int = None,
    win_length: int = None,
) -> np.ndarray:
    """
    Compute per-frame loudness by A-weighting each FFT bin of the magnitude
    spectrogram, summing the weighted bins within a frame, and taking the RMS.

    This mirrors the Sketch2Sound control signal: 5 ms hops, mono audio,
    linear amplitude weighting derived from the A-weighting curve, and an RMS
    envelope that can be aligned directly to the VAE latent frame rate.
    """
    if hop_length is None:
        hop_length = int(sr / 200.0)  # 5 ms step
    if n_fft is None:
        n_fft = 1024
    if win_length is None:
        win_length = n_fft

    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim == 2:
        audio_np = librosa.to_mono(audio_np)

    S = np.abs(
        librosa.stft(
            audio_np,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    A_weight_amp = librosa.db_to_amplitude(librosa.A_weighting(freqs))
    weighted = S * A_weight_amp[:, np.newaxis]
    loudness = np.sqrt(np.maximum(np.sum(weighted**2, axis=0), 1e-12))
    return loudness


def spectral_centroid(
    y: np.ndarray,
    sr: int,
    hop_length: int = None,
    n_fft: int = None,
    win_length: int = None,
    midi: bool = True,
) -> np.ndarray:
    """
    Compute the spectral centroid per frame and convert it to MIDI values (semitones),
    scaled to roughly (0, 1) by dividing by 127.

    This center of mass directly correlates with perceived brightness or timbre:
    High spectral centroid: Energy concentrated in higher frequencies = brighter/sharper sound
    Low spectral centroid: Energy concentrated in lower frequencies = darker/mellower sound

    """
    if hop_length is None:
        hop_length = int(sr / 200.0)  # 5ms per frame
    if n_fft is None:
        n_fft = 1024  # default FFT size
    if win_length is None:
        win_length = n_fft  # default window length

    centroid_hz = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, win_length=win_length
    )[0]
    # Ensure no zero values for safe logarithm calculation
    centroid_hz = np.maximum(centroid_hz, 1e-6)
    if midi:
        centroid_midi = 69 + 12 * np.log2(centroid_hz / 440.0)
        # scale to roughly (0, 1) by dividing by 127
        centroid_midi = centroid_midi / 127.0
        return centroid_midi
    return centroid_hz


def extract_pitch_probability(
    audio: np.ndarray | torch.Tensor,
    sr: int,
    hop_length: int = None,
    model: str = "tiny",
    device: str = "cuda:0",
) -> np.ndarray:
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
        hop_length = int(sr / 200.0)

    # torchcrepe.preprocess returns a generator of batches; get the first batch.
    batch = next(torchcrepe.preprocess(audio, sr, hop_length))
    batch = batch.to(device)
    with torch.no_grad():
        # Call torchcrepe.infer with proper parameters as per the repository
        pitch_probs = torchcrepe.infer(batch, model=model, device=device)

    # Zero out all probabilities below 0.1 to avoid leaking timbral information
    pitch_probs = torch.where(
        pitch_probs < 0.1, torch.zeros_like(pitch_probs), pitch_probs
    )
    pitch_probs = pitch_probs.cpu()
    torch.cuda.empty_cache()
    return pitch_probs.numpy()




if __name__ == "__main__":
    # test the functions
    example_audio = "example_audio/in.wav"

    y, sr = librosa.load(example_audio, sr=None)
    loudness = compute_loudness(y, sr)
    centroid = spectral_centroid(y, sr)
    pitch_probs = extract_pitch_probability(y, sr)
    # print shapes
    print("Audio shape:", y.shape)
    print("Sample Rate:", sr)
    print("Loudness shape:", loudness.shape)
    print("Spectral Centroid shape:", centroid.shape)
    print("Pitch Probability shape:", pitch_probs.shape)
