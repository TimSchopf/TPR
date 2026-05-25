import os
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import argparse

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

cache_dir="../models/"
output_dir = Path("../data/pt/")
output_dir.mkdir(parents=True, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_DISABLE_XET'] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = "..."
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# check available GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# load data
ds = load_dataset("TimSchopf/RINoBench")
print(ds)
labels = load_dataset("TimSchopf/RINoBench", "class_descriptions")
label_descriptions = [str(l['label'])+": "+l['description'] for l in labels['class_descriptions']]

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

# tokenizer and model
model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, dtype=dtype, trust_remote_code=True, device_map="auto")
print("Model used for prediction:", model_id)

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

# ---------------------------
# Prompt Builder
# ---------------------------
def build_ai_researcher_prompt(research_idea: dict, related_works: list, class_descriptions: list) -> str:
    # Join class_descriptions list into a bullet point string, each on a new line with indentation
    class_desc_str = "\n       - " + "\n       - ".join(class_descriptions)

    return f'''You are a professor specialized in machine learning. You are given a research idea and related works and you need to decide whether the idea is creative and different from existing works on the topic, and brings fresh insights. You should consider all related works when judging the novelty.

    The research idea is:
    {research_idea}

    We have found these related works:
    {related_works}

    Decide on the novelty of the idea on a scale of 1 to 5.
    {class_desc_str}

    Give a short justification for your score. If you give a low score, you should specify similar related works. (Your rationale should be at least 2-3 sentences.)

    Respond in the following format:
    ```json
    {{
      "novelty_score": <1|2|3|4|5>,
      "justification": "<short justification>"
    }}
    ```
    '''

# test generation
x = 0
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
messages = [{
            "role": "user",
            "content": build_ai_researcher_prompt(
            research_idea=ds['test']['research_idea'][i],
            related_works=ds['test']['related_works'][i],
            class_descriptions=label_descriptions
        )
        }]

r = pipe(messages, max_new_tokens=5000)
print(r[0]['generated_text'][-1]['content'])


# -----------------------------------------------------
#  RESUME LOGIC — Identify completed indices
# -----------------------------------------------------
def get_completed_indices(approach_name, split, model_id):
    prefix = f"{approach_name}_{split}_{model_id.replace('/', '-')}_"
    done = []
    for file in output_dir.iterdir():
        if file.name.startswith(prefix) and file.name.endswith(".pt"):
            idx = int(file.stem.split("_")[-1])
            done.append(idx)
    return set(done)

# -----------------------------------------------------
#   PROCESS SPLIT (TRAIN or TEST) WITH RESUME
# -----------------------------------------------------
def run_split(split_name, data_split, approach_name):

    completed = get_completed_indices(approach_name, split_name, model_id)
    total = len(data_split)

    print(f"\n=== Processing {split_name.upper()} ===")
    print(f"Samples already completed: {len(completed)}")

    all_indices = list(range(total))
    missing_indices = [i for i in all_indices if i not in completed]

    if len(missing_indices) == 0:
        print("Nothing left to do. DONE.")
        return

    print(f"Missing samples: {len(missing_indices)}")
    print(f"First missing index: {missing_indices[0]}")

    for i in tqdm(missing_indices, desc=f"Predict {split_name}"):
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map='auto')
        messages = [{
            "role": "user",
            "content": build_ai_researcher_prompt(
            research_idea=ds['test']['research_idea'][i],
            related_works=ds['test']['related_works'][i],
            class_descriptions=label_descriptions
        )
        }]

        parsed_outputs = pipe(messages, max_new_tokens=5000)[0]['generated_text'][-1]['content']

        outfile = output_dir / f"{approach_name}_{split_name}_{model_id.replace('/', '-')}_{i}.pt"
        torch.save(
            {
                "parsed_outputs": parsed_outputs
            },
            outfile
        )

        del parsed_outputs
        torch.cuda.empty_cache()

# -----------------------------------------------------
#                     RUN
# -----------------------------------------------------
run_split("test", ds["test"], "ai_researcher")