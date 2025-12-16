import torch
import torchaudio

class Reverb(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        room_size_min=1,
        room_size_max=5,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max
        self.rir, sr = torchaudio.load('data/RIR.wav')
        self.rir = torchaudio.functional.resample(self.rir, sr, self.sample_rate)

    def forward(self, audio):
        # Pick a random room size and random locations
        room = torch.randint(self.room_size_min, self.room_size_max + 1, (3,))
        source = torch.zeros(3)
        for i in range(3):
            source[i] = torch.randint(room[i], (1,))
        mic = torch.zeros(3)
        for i in range(3):
            mic[i] = torch.randint(room[i], (1,))

        return torchaudio.functional.fftconvolve(audio, self.rir)