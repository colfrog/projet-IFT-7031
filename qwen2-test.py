import torch
import torchaudio
import json
import librosa
from datasets import Dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel
from miditok import REMI, TokenizerConfig
import mido
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR = "./qwen2-audio-finetuned-v4"
ADAPTER_PATH = OUTPUT_DIR
MODEL_PATH = Path("/home/lcimon/scratch/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a/")
DATA_PATH = Path("/home/lcimon/scratch/training_data")
MIDI_PATHS = DATA_PATH.glob("**/*.mid")
SAMPLE_PATHS = DATA_PATH.glob("*/*")
REMI_PATH = "remi.json"
MAX_LENGTH = 8192

print(f"CUDA Version: {torch.version.cuda}")

print("Loading model and processor... ", end="")
print(f"({MODEL_PATH})" if os.path.exists(MODEL_PATH) else f"({MODEL_ID})")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_ID)
print(f"Sampling rate: {processor.feature_extractor.sampling_rate}")

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    #MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_ID,
    ADAPTER_PATH,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="eager"
)

class RemiCompactor():
    def __init__(self):
        if os.path.exists(REMI_PATH):
            self.remi = REMI(params=Path(REMI_PATH))
        else:
            remi_config = TokenizerConfig(
                num_velocities=16,
                use_pitch_bends=True,
                pitch_bend_range=(-8192, 8191, 1024), # We need a lot of values for pitch bends
                use_control_changes=True,
                use_programs=True,
                one_token_stream_for_programs=False,
                use_time_signatures=True,
                ac_polyphony_track=True
            )
            self.remi = REMI(remi_config)
            self.remi.train(vocab_size=30000, files_paths=MIDI_PATHS + MUSICNET_MIDI_PATHS)
            self.remi.save(REMI_PATH)

    def explode_mpe_to_tracks(self, input_midi_path, output_midi_path):
        """
        Splits an MPE MIDI file (usually Type 0 or 1 merged) into
        separate tracks for each Channel (1-16).
        This forces the Tokenizer to treat them as separate 'Voices'.
        """
        mid = mido.MidiFile(input_midi_path)
        new_mid = mido.MidiFile()
        new_mid.ticks_per_beat = mid.ticks_per_beat

        # Create 17 empty tracks (0=Global, 1-16=Channels)
        channels = [mido.MidiTrack() for _ in range(17)]

        # Sort every message into its corresponding track based on channel
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, 'channel'):
                    # Channels are 0-15 in mido, mapped to tracks 1-16
                    channels[msg.channel + 1].append(msg)
                else:
                    # Meta messages (Tempo, Time Sig) go to Track 0 (Global)
                    channels[0].append(msg)

        # Add non-empty tracks to the new file
        for track in channels:
            if len(track) > 0:
                new_mid.tracks.append(track)

        new_mid.save(output_midi_path)
        #print(f"Exploded MPE file to {len(new_mid.tracks)} tracks.")

    def midi_to_str(self, path, mpe=False):
        if mpe:
            self.explode_mpe_to_tracks(path, "/tmp/exploded.mid")
            path = "/tmp/exploded.mid"
        tokens = self.remi.encode(path)
        if isinstance(tokens, list):
            tokens = tokens[0]

        self.remi.complete_sequence(tokens)
        return " ".join(tokens.tokens)

print("Processing dataset...")
midi_compactor = RemiCompactor()
audio_path = "output/other.wav"
midi_path = "output/other.mid"

tokenized_midi = midi_compactor.midi_to_str(midi_path)
tokenized_midi = tokenized_midi[:len(tokenized_midi)//2]

audio, sr = torchaudio.load(audio_path)
audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=processor.feature_extractor.sampling_rate)
audio = torch.mean(audio, dim=0)
audio = audio.numpy().squeeze()
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": f"Convert the audio to MPE MIDI\n\n{tokenized_midi}"}
        ]
    }
]

formatted_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=formatted_text, audio=audio, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate)
for key in inputs:
    inputs[key] = inputs[key].cuda()

generate_ids = model.generate(**inputs, max_length=MAX_LENGTH)
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)
