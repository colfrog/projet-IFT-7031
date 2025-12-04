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

print("Loading dataset...")
raw_data = load_data(DATA_FILE)

print("Processing dataset...")
audios = []
text_prompts = []
instruction_lens = []
count = 0
for sample in enumerate(raw_data):
    print(f"\r{count}/{len(raw_data)}", end="")
    audio, _ = librosa.load(sample['audio'], sr=processor.feature_extractor.sampling_rate)
    audios.append(audio)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": sample['audio']},
                {"type": "text", "text": sample['instruction']}
            ]
        },
        {
            "role": "assistant",
            "content": sample['output']
        }
    ]

    instruction_lens.append(len(processor.apply_chat_template([messages[0]], add_generation_prompt=False, tokenize=True)))
    formatted_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    text_prompts.append(formatted_text)
    count += 1
print()

inputs = processor(text=text_prompts, audio=audios, return_tensors="pt", padding=True)
inputs["labels"] = inputs["input_ids"].clone()
padding_mask = inputs["input_ids"] == processor.tokenizer.pad_token_id
inputs["labels"][padding_mask] = -100

# Ignore the user input to avoid learning it.
for i, prompt_len in enumerate(instruction_lens):
    inputs["labels"][i, :prompt_len] = -100

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
