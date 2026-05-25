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
ds = load_dataset(".../RINoBench")
print(ds)
labels = load_dataset(".../RINoBench", "class_descriptions")
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
# Prompt Builder for the AI Scientist (https://arxiv.org/abs/2408.06292)
# ---------------------------
class_desc_str = "\n       - " + "\n       - ".join(label_descriptions)
ai_scientist_system_prompt = f"""You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given the top search results from the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top search results will be presented to you with the abstracts.

Decide on the novelty of the idea on a scale of 1 to 5.
{class_desc_str}
"""

def build_ai_scientist_prompt(research_idea: dict, related_works: list) -> str:
    return f'''You have this idea:

    """
    {research_idea}
    """

    The top search results are:
    """
    {related_works}
    """

    Respond in the following format:
    ```json
    {{
      "THOUGHT": <THOUGHT>,
      "NOVELTY": <1|2|3|4|5>
    }}
    ```

    In <THOUGHT>, briefly reason over the idea and identify any aspect that could help you make your decision.

    In <NOVELTY>, respond with ONLY an integer between 1-5, corresponding to the novelty of the idea.
    '''

# test generation
x = 0
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
messages = [{
            "role": "system",
            "content": ai_scientist_system_prompt
        },
        {
            "role": "user",
            "content": build_ai_scientist_prompt(
            research_idea=ds['test']['research_idea'][i],
            related_works=ds['test']['related_works'][i]
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
            "role": "system",
            "content": ai_scientist_system_prompt
        },
        {
            "role": "user",
            "content": build_ai_scientist_prompt(
            research_idea=ds['test']['research_idea'][i],
            related_works=ds['test']['related_works'][i],
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
run_split("test", ds["test"], "ai_scientist")