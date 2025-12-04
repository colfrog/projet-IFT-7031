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
import os

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DATA_FILE = "train_data.json"
OUTPUT_DIR = "./qwen2-audio-finetuned"
MODEL_PATH = os.path.abspath("/home/lcimon/scratch/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a/")

print("Loading model and processor... ", end="")
print(f"({MODEL_PATH})" if os.path.exists(MODEL_PATH) else f"({MODEL_ID})")

processor = AutoProcessor.from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2" # Use "eager" if flash attn not installed
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

# Define LoRA Config
peft_config = LoraConfig(
    r=64, # Rank
    lora_alpha=16,
    target_modules="all_linear",
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

print("Loading dataset...")
raw_data = load_data(DATA_FILE)

print("Processing dataset...")
audios = []
for sample in raw_data:
    audio, _ = librosa.load(sample.pop('audio'), sr=processor.feature_extractor.sampling_rate)
    audios.append(audio)

text = processor.apply_chat_templates(raw_data, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
dataset = Dataset.from_dict(inputs)

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
)

# --- 6. Train ---
print("Starting training...")
trainer.train()

print(f"Training finished. Saving to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
