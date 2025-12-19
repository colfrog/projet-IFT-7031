import torch
import torchaudio
from torchaudio_augmentations import Compose, RandomApply, PolarityInversion, Noise, Gain, Reverb, HighLowPass
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
from sklearn.metrics import recall_score, precision_score
import mido
import os
from pathlib import Path

import kagglehub
MUSICNET_PATH = "/home/lcimon/scratch/kaggle/datasets/imsparsh/musicnet-dataset/versions/1"
if not os.path.exists(MUSICNET_PATH):
    MUSICNET_PATH = kagglehub.dataset_download("imsparsh/musicnet-dataset")
    print(f"MusicNet: {MUSICNET_PATH}")
MUSICNET_PATH = Path(MUSICNET_PATH)
MUSICNET_AUDIO_PATHS = list(MUSICNET_PATH.joinpath("musicnet/musicnet/train_data").glob("*.wav"))
MUSICNET_MIDI_PATHS = []
for audio_path in MUSICNET_AUDIO_PATHS:
    glob = MUSICNET_PATH.joinpath("musicnet_midis/musicnet_midis").glob(f"*/{audio_path.stem}*.mid")
    for midi_path in glob:
        MUSICNET_MIDI_PATHS.append(midi_path)

assert len(MUSICNET_AUDIO_PATHS) == len(MUSICNET_MIDI_PATHS)

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR = "./qwen2-audio-finetuned-v4"
REMI_PATH = "remi.json"
MODEL_PATH = Path("/home/lcimon/scratch/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a/")
DATA_PATH = Path("/home/lcimon/scratch/training_data")
MIDI_PATHS = list(DATA_PATH.glob("**/*.mid"))
SAMPLE_PATHS = list(DATA_PATH.glob("*/*"))

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
    dtype=torch.float16,
    quantization_config=bnb_config,
    attn_implementation="sdpa"
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
MAX_FRAMES = 600000
ignored = 0
### MusicNet files are too big to train on, this should be explored in further works
#for i in range(len(MUSICNET_AUDIO_PATHS)):
    #metadata = torchaudio.info(MUSICNET_AUDIO_PATHS[i])
    #if metadata.num_frames > MAX_FRAMES: # Ignore large audio files to avoid running out of memory
        #ignored += 1
        #continue
    #audio_paths.append(str(MUSICNET_AUDIO_PATHS[i]))
    #midi_paths.append(str(MUSICNET_MIDI_PATHS[i]))
    #mpe_paths.append("")

#print(f"{ignored} samples ignored due to size")

audios = []
text_prompts = []
instruction_lens = []
paths = list(SAMPLE_PATHS)
for count, path in enumerate(paths):
    print(f"\r{count + 1}/{len(paths)}", end='')
    midi_path = f"{path}/plain.mid"
    mpe_path = f"{path}/mpe.mid"
    audio_path = f"{path}/audio.wav"

    tokenized_midi = midi_compactor.midi_to_str(midi_path)
    tokenized_mpe = midi_compactor.midi_to_str(mpe_path, mpe=True)

    audio, sr = torchaudio.load(audio_path, format='wav')
    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=processor.feature_extractor.sampling_rate)
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
                {"type": "text", "text": f"Convert this audio to MPE MIDI\n\n{tokenized_midi}"}
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

    prompt = processor.apply_chat_template([messages[:2]], add_generation_prompt=False, tokenize=False)
    prompt_tokens = processor.tokenizer(prompt, add_special_tokens=False)
    instruction_lens.append(len(prompt_tokens))
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    text_prompts.append(text)
print()

# We may add more later
augments = [
    RandomApply([PolarityInversion()], p=0.5),
    RandomApply([Noise(min_snr=0.1, max_snr=0.5)], p=0.6),
    RandomApply([Reverb(processor.feature_extractor.sampling_rate)], p=0.5), # It's not random or exaggerated but it's better than nothing
    RandomApply([Gain(min_gain=-10, max_gain=10)], p=0.9),
    RandomApply([HighLowPass(processor.feature_extractor.sampling_rate)], p=0.5)
]
transform = RandomApply([Compose(augments)], p=0.8) # We want 20% of the samples to not have augmentations.

def data_collator(features):
    audios = [f["audio"] for f in features]
    text_prompts = [f["text"] for f in features]
    instruction_lens = [f["len"] for f in features]
    eval = [f["eval"] for f in features]

    processed_audios = []
    for i in range(len(audios)):
        audio = torch.tensor(audios[i])
        audio = transform(audio) # Apply transformations
        audio = torch.mean(audio, dim=0)  # Convert to mono
        audio = audio.numpy().squeeze() # Make sure we have a 1D numpy array
        processed_audios.append(audio)

    inputs = processor(text=text_prompts, audio=processed_audios, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate)
    inputs["labels"] = inputs["input_ids"].clone()
    padding_mask = inputs["input_ids"] == processor.tokenizer.pad_token_id
    inputs["labels"][padding_mask] = -100

    # Ignore the user input to avoid learning it.
    for i, prompt_len in enumerate(instruction_lens):
        inputs["labels"][i, :prompt_len] = -100

    return inputs

dataset = Dataset.from_dict({"audio": audios, "text": text_prompts, "len": instruction_lens, "eval": [False]*len(audios)})
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 10, 10])
print("Eval samples:", paths[eval_dataset.indices])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    labels_flat = labels.flatten()
    preds_flat = preds.flatten()
    mask = labels_flat != -100
    labels = labels_flat[mask]
    preds = preds_flat[mask]

    accuracy = (labels == preds).mean()
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=10,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="epoch",
    eval_accumulation_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print(f"Training finished. Saving to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
