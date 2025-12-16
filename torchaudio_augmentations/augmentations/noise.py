import random
import torch


class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio).item()
        noise_std = random.uniform(self.min_snr*std, self.max_snr*std)
        noise = (noise_std*torch.randn(audio.shape, dtype=torch.float32, device=audio.device))

        return audio + noise