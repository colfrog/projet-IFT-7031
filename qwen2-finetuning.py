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
import os

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR = "./qwen2-audio-finetuned"
MODEL_PATH = os.path.abspath("/home/lcimon/scratch/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a/")
DATA_PATH = os.path.abspath("/home/lcimon/scratch/training_data")
MIDI_PATHS = DATA_PATH.glob("**/*.mid")
SAMPLE_PATHS = DATA_PATH.glob("*/*")

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
    attn_implementation="flash_attention_2"
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
        remi_config = TokenizerConfig(num_velocities=16, use_chords=True, use_tempos=True, beat_res={(0, 4): 8, (4, 12): 4})
        self.remi = REMI(remi_config)
        self.remi.train(vocab_size=30000, files_paths=MIDI_PATHS)

    def midi_to_str(self, path):
        try:
            tokens = self.remi.encode(path)
            self.remi.complete_sequence(tokens)
            return " ".join(tokens.tokens)
        except Exception as e:
            print(f"Error converting {midi_path}: {e}")
            return ""

print("Processing dataset...")
audios = []
text_prompts = []
instruction_lens = []
midi_compactor = RemiCompactor()
for count, path in enumerate(SAMPLE_PATHS):
    print(f"\r{count}/{len(MIDI_PATHS)}", end="")

    audio_path = f"${path}/audio.wav"
    midi_path = f"${path}/plain.mid"
    mpe_path = f"${path}/mpe.mid"

    audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate, mono=True)
    audios.append(audio)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": midi_compactor.midi_to_str(midi_path)}
            ]
        },
        {
            "role": "assistant",
            "content": midi_compactor.midi_to_str(mpe_path)
        }
    ]

    instruction_lens.append(len(processor.apply_chat_template([messages[0]], add_generation_prompt=False, tokenize=True)))
    formatted_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    if count == 1:
        print(formatted_text)
    text_prompts.append(formatted_text)
print()

def data_collator(features):
    text = [f["text"] for f in features]
    audio = [f["audio"] for f in features]
    lens = [f["instruction_len"] for f in features]

    inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate)
    inputs["labels"] = inputs["input_ids"].clone()
    padding_mask = inputs["input_ids"] == processor.tokenizer.pad_token_id
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
    num_train_epochs=1,
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
