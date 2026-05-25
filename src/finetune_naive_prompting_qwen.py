print("Running training script...")
import torch
from datasets import load_dataset
import os
from pathlib import Path
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model, TaskType


# -----------------------------------------------------
#                Command Line ARGUMENTS
# -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    required=True,          # must be provided
    help="Hugging Face model ID"
)
args = parser.parse_args()

# -----------------------------------------------------
#                CONFIG / SETUP
# -----------------------------------------------------


cache_dir="/home/tisc207h/workspaces/horse/tisc207h-ResearchNoveltyJudgment/research_novelty_judgment/models/"
output_dir = Path("../data/pt/")
output_dir.mkdir(parents=True, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_DISABLE_XET'] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = "..."

# check available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("No GPU available")
else:
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# load data
ds = load_dataset("TimSchopf/RINoBench")
labels = load_dataset("TimSchopf/RINoBench", "class_descriptions")
label_descriptions = [str(l['label'])+": "+l['description'] for l in labels['class_descriptions']]

model_id = args.model_id

# get compute capability
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    if torch.cpu.is_available() and hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, dtype=dtype, trust_remote_code=True)
print("loaded model:", model.config._name_or_path)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    for device_id in range(num_devices):
        print(f"Device {device_id}: {torch.cuda.get_device_name(device_id)}")
        print(f"  Memory Allocated (GB): {torch.cuda.memory_allocated(device_id) / 1024**3:.2f}")
        print(f"  Memory Reserved (GB): {torch.cuda.memory_reserved(device_id) / 1024**3:.2f}")
        free, total = torch.cuda.mem_get_info(device_id)
        print(f"  Free Memory (GB): {free / 1024**3:.2f}")
        print(f"  Total Memory (GB): {total / 1024**3:.2f}")
else:
    print("CUDA is not available.")

# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()

# ---------------------------
# Prompt Builder
# ---------------------------

def build_judgment_prompt(research_idea: dict, related_works: list, class_descriptions: list) -> str:
    # Join class_descriptions list into a bullet point string, each on a new line with indentation
    class_desc_str = "\n       - " + "\n       - ".join(class_descriptions)

    related_works = [{'title':r['title'], 'abstract':r['abstract']} for r in related_works]

    return f"""
    You are an expert in machine learning research evaluation. You will be given two inputs:

    1. A research idea with objective, problem statement, and solution approach.
    2. A list of related works, each with a title and abstract.

    Your task is to **assess the novelty of the research idea** compared to the related works.

    ### Instructions:
    - Analyze the research idea and summarize its key contributions.
    - Compare it with the related works to identify overlaps and differences.
    - Specifically, assess whether the idea introduces **significant new aspects** not present in existing work, or if it is largely a variation on known approaches.
    - Provide your output as a **JSON object only**, with:
      - `"reasoning"`: a short paragraph (2–4 sentences) explaining the reasoning behind the novelty score.
      - `"novelty_score"`: an integer between 1-5 where: {class_desc_str}

    ### Inputs:

    **Research Idea:**
    {research_idea}

    **Related Works:**
    {related_works}

    ### Output Format:
    ```json
    {{
      "reasoning": "<short explanation>",
      "novelty_score": <1|2|3|4|5>
    }}
    ```
    """

def format_example(example):
    
    system_prompt = "You are an expert researcher experienced in judging the novelty of a research idea."
    user_prompt = build_judgment_prompt(
            research_idea=example['research_idea'],
            related_works=example['related_works'][:10], # limit to first n related works to fit context window
            class_descriptions=label_descriptions
    )
    assistant_response = f"""```json
    {{
      "reasoning": "{example['novelty_reasoning']}",
      "novelty_score": {example['novelty_score']}
    }}
    ```"""
    
    # LLM chat template format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]
    
    # Apply LLM chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, reasoning_effort="low")
    return {"text": text}

# apply chat templates
train_dataset = ds["train"].map(format_example)
test_dataset = ds["test"].map(format_example)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"])

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    #output_dir=cache_dir+model_id.split('/')[1]+"-novelty-lora",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, # This fixes the "different number of tensors" error.
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=20,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    remove_unused_columns=False,
    deepspeed=None,
    fsdp="full_shard auto_wrap",
    fsdp_config={
        # This line is the FIX for the "Could not find the transformer layer class" error
        "transformer_layer_cls_to_wrap": "Qwen3DecoderLayer", 
        # We can remove "activation_checkpointing": True here because 
        # gradient_checkpointing=True above handles it when combined with HF Trainer
    },
    dataloader_pin_memory=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# Train the model
trainer.train()

# Save the LoRA adapter
trainer.save_model(cache_dir+model_id.split('/')[1]+"-novelty-lora-final")
tokenizer.save_pretrained(cache_dir+model_id.split('/')[1]+"-novelty-lora-final")

print("Finished training succesfully.")