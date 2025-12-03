import torch
import json
import librosa
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from datasets import Dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DATA_FILE = "train_data.json"
OUTPUT_DIR = "./qwen2-audio-finetuned"

print("Loading model and processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
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


raw_data = load_data(DATA_FILE)
dataset = Dataset.from_list(raw_data)


def format_audio_and_text(data):
    """
    Loads audio and formats the chat template.
    """
    # Load Audio
    audio_path = data['audio']
    # Qwen2-Audio generally expects 16kHz
    audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

    # Format the Conversation
    # Qwen2-Audio uses specific tags. We rely on the processor to handle them via chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},  # Processor handles the dummy URL
                {"type": "text", "text": data['instruction']}
            ]
        },
        {
            "role": "assistant",
            "content": data['output']
        }
    ]

    # We return the raw speech array and the message structure
    # The actual tokenization happens in the DataCollator to allow dynamic padding
    return {
        "audio_array": audio,
        "messages": messages
    }


print("Processing dataset...")
dataset = dataset.map(format_audio_and_text, remove_columns=dataset.column_names)


# --- 3. Custom Data Collator ---

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        feature_values_list = []

        for feature in features:
            audio = feature["audio_array"]
            messages = feature["messages"]

            # 1. Apply Chat Template to get text input
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

            # 2. Process inputs (Audio + Text)
            # This converts audio to features and text to tokens
            inputs = self.processor(
                text=[text],
                audios=[audio],
                return_tensors="pt",
                padding=False
            )

            # 3. Create Labels (Masking User Input)
            # We must calculate where the assistant's response starts to mask the rest for loss calculation
            input_ids = inputs["input_ids"][0]
            labels = input_ids.clone()

            # Find the split between user prompt and assistant response
            # Note: This is a simplified masking strategy. For strict instruction tuning,
            # you usually mask everything up to the last <|im_start|>assistant header.
            # Qwen uses specific tokens.

            # For simplicity in this script, we train on the whole sequence or rely on the
            # model's internal handling, but typically we mask non-response tokens with -100.
            # A simple heuristic: Mask everything before the last turn's text start.

            # (Optional: Advanced masking logic would go here.
            # For now, we train on the sequence provided by the processor which is standard for many SFT tasks)

            input_ids_list.append(input_ids)
            feature_values_list.append(inputs["input_features"][0])
            labels_list.append(labels)  # Currently training on prompt+response (standard for some causal LM setups)

        # Padding
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100  # -100 is ignored by CrossEntropyLoss
        )

        # Audio features also need padding/stacking
        # Qwen2Audio processor usually returns list of tensors for features, we stack them
        batch_input_features = torch.nn.utils.rnn.pad_sequence(
            feature_values_list, batch_first=True, padding_value=0.0
        )

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "input_features": batch_input_features,
            # Attention mask is automatically created by the model based on pad tokens usually,
            # but we can explicitely create it:
            "attention_mask": batch_input_ids.ne(self.processor.tokenizer.pad_token_id)
        }


data_collator = DataCollatorSpeechSeq2Seq(processor=processor)

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
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

# --- 6. Train ---
print("Starting training...")
trainer.train()

print(f"Training finished. Saving to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)