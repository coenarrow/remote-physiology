import torch
import numpy as np
import cv2

def temporal_fft(video):
    """
    Convert uint8 video to frequency domain using FFT along temporal dimension.
    
    Args:
        video: torch.Tensor of shape (N, H, W) with dtype uint8
    
    Returns:
        freq: torch.Tensor of shape (N//2 + 1, H, W) with dtype complex64
    """
    return torch.fft.rfft(video.float(), dim=0, norm='ortho')

def temporal_ifft(freq, n_frames):
    """
    Convert frequency domain back to time domain.
    
    Args:
        freq: torch.Tensor of shape (N//2 + 1, H, W) with dtype complex64
        n_frames: int, original number of frames
    
    Returns:
        video: torch.Tensor of shape (N, H, W) with dtype uint8
    """
    video_float = torch.fft.irfft(freq, n=n_frames, dim=0, norm='ortho')
    return video_float.round().clamp(0, 255).to(torch.uint8)

# Create Gaussian sine wave video
N, H, W = 100, 480, 640

# Temporal sine wave
t = torch.linspace(0, 2 * np.pi, N).view(N, 1, 1)
temporal_wave = torch.sin(5 * t)

# Spatial Gaussian
y = torch.linspace(-1, 1, H).view(1, H, 1)
x = torch.linspace(-1, 1, W).view(1, 1, W)
spatial_gaussian = torch.exp(-(x**2 + y**2) / 0.3)

# Combine
video_float = 127.5 + 127.5 * temporal_wave * spatial_gaussian
video = video_float.round().clamp(0, 255).to(torch.uint8)

print(f"Original size: {video.numel() * video.element_size() / 1e6:.2f} MB")

# Transform to frequency domain
freq = temporal_fft(video)
print(f"Data type in frequency domain: {freq.dtype}")
print(f"Frequency domain size: {freq.numel() * freq.element_size() / 1e6:.2f} MB")

# Reconstruct
video_reconstructed = temporal_ifft(freq, N)
max_error = (video.float() - video_reconstructed.float()).abs().max()
print(f"Max reconstruction error: {max_error:.6f}")

# Display with OpenCV
print("\nDisplaying original video (press 'q' to quit, 'r' to restart)")
while True:
    for i in range(N):
        frame = video[i].numpy()
        cv2.imshow('Original Gaussian Sine Wave', frame)
        
        key = cv2.waitKey(30)  # ~30 fps
        if key == ord('q'):
            break
        elif key == ord('r'):
            break
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Optionally display reconstructed video
print("\nDisplaying reconstructed video (press 'q' to quit)")
for i in range(N):
    frame = video_reconstructed[i].numpy()
    cv2.imshow('Reconstructed from Frequency Domain', frame)
    
    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()