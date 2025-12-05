import torch
import json
import librosa
from datasets import Dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from miditok import REMI, TokenizerConfig
import mido
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR = "./qwen2-audio-finetuned-v2"
MODEL_PATH = Path("/home/lcimon/scratch/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a/")
DATA_PATH = Path("/home/lcimon/scratch/training_data")
MIDI_PATHS = DATA_PATH.glob("**/*.mid")
SAMPLE_PATHS = DATA_PATH.glob("*/*")
MAX_SIZE = 8192

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
    MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="eager"
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

# Define LoRA Config
peft_config = LoraConfig(
    r=64, # Rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

class RemiCompactor():
    def __init__(self):
        remi_config = TokenizerConfig(
            num_velocities=16,
            use_pitch_bends=True,
            use_control_changes=True,
            use_programs=True,
            one_token_stream_for_programs=False,
            use_time_signatures=True,
            ac_polyphony_track=True
        )
        self.remi = REMI(remi_config)
        self.remi.train(vocab_size=30000, files_paths=list(MIDI_PATHS))

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
audios = []
text_prompts = []
instruction_lens = []
midi_compactor = RemiCompactor()
paths = list(SAMPLE_PATHS)
for count, path in enumerate(paths):
    print(f"\r{count}/{len(paths)}", end="")

    audio_path = f"{path}/audio.wav"
    midi_path = f"{path}/plain.mid"
    mpe_path = f"{path}/mpe.mid"

    tokenized_midi = midi_compactor.midi_to_str(midi_path)
    tokenized_mpe = midi_compactor.midi_to_str(mpe_path, mpe=True)

    audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate, mono=True)
    audios.append(audio)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": f"Convert the following to MPE according to the audio\n\n{tokenized_midi}"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": tokenized_mpe
                }
            ]
        }
    ]

    instruction_lens.append(len(processor.apply_chat_template([messages[0]], add_generation_prompt=False, tokenize=True)))
    formatted_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    text_prompts.append(formatted_text)

    print(f"\r{count}/{len(paths)}", end="")
print()

def data_collator(features):
    text = [f["text"] for f in features]
    audio = [f["audio"] for f in features]
    lens = [f["instruction_len"] for f in features]

    inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate)
    inputs["labels"] = inputs["input_ids"].clone()
    padding_mask = inputs["input_ids"] == processor.tokenizer.pad_token_id
    inputs["attention_mask"] = padding_mask == False
    inputs["labels"][padding_mask] = -100

    # Ignore the user input to avoid learning it.
    for i, prompt_len in enumerate(lens):
        inputs["labels"][i, :prompt_len] = -100

    return inputs

dataset = Dataset.from_dict({"text": text_prompts, "audio": audios, "instruction_len": instruction_lens})

# --- 4. Training Arguments ---

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False
)

# --- 5. Trainer ---

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator
)

# --- 6. Train ---
print("Starting training...")
trainer.train()

print(f"Training finished. Saving to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
