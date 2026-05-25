import os
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import argparse
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
)

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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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

# harmony encoder
harmony_encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

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


# -----------------------------------------------------
#                Prompt
# -----------------------------------------------------
example_1 = ds["train"].filter(lambda ex: ex["source"] == 'https://openreview.net/forum?id=Wac06sAkHk')[0]
relevant_related_works = ["A Continual Learning Survey: Defying Forgetting in Classification Tasks","Task-agnostic Continual Learning with Hybrid Probabilistic Models"]
relevant_related_works = [w.lower() for w in relevant_related_works]
rel_works = []
for w in example_1['related_works']:
    if w['title'].lower() in relevant_related_works:
        rel_works.append(w)
if len(set([w['title'] for w in rel_works])) == len(set(relevant_related_works)):
    print("Check:", True)
else:
    print("Check:", False)
    print("Not parsed related works:",set(relevant_related_works) - set([w['title'].lower() for w in rel_works]))

example_1['related_works'] = rel_works

example_2 = ds["train"].filter(lambda ex: ex["source"] == 'https://openreview.net/forum?id=zYWtq_HUCoi')[0]
relevant_related_works = ["Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon","Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning"]
relevant_related_works = [w.lower() for w in relevant_related_works]
rel_works = []
for w in example_2['related_works']:
    if w['title'].lower() in relevant_related_works:
        rel_works.append(w)
if len(set([w['title'] for w in rel_works])) == len(set(relevant_related_works)):
    print("Check:", True)
else:
    print("Check:", False)
    print("Not parsed related works:",set(relevant_related_works) - set([w['title'].lower() for w in rel_works]))

example_2['related_works'] = rel_works

example_3 = ds["train"].filter(lambda ex: ex["source"] == 'https://openreview.net/forum?id=zlwBI2gQL3K')[0]
relevant_related_works = ["Contrastive Multi-View Representation Learning on Graphs","Learning Entity and Relation Embeddings for Knowledge Graph Completion"]
relevant_related_works = [w.lower() for w in relevant_related_works]
rel_works = []
for w in example_3['related_works']:
    if w['title'].lower() in relevant_related_works:
        rel_works.append(w)
if len(set([w['title'] for w in rel_works])) == len(set(relevant_related_works)):
    print("Check:", True)
else:
    print("Check:", False)
    print("Not parsed related works:",set(relevant_related_works) - set([w['title'].lower() for w in rel_works]))

example_3['related_works'] = rel_works

example_4 = ds["train"].filter(lambda ex: ex["source"] == 'https://openreview.net/forum?id=vuD2xEtxZcj')[0]
example_4['related_works'] = example_4['related_works'][:2]

example_5 = ds["train"].filter(lambda ex: ex["source"] == 'https://openreview.net/forum?id=zEn1BhaNYsC')[0]
example_5['related_works'] = example_4['related_works'][:2]

few_shot_examples = [example_1, example_2, example_3, example_4, example_5]

def built_novelty_cot_prompt(idea, similar_documents, few_shot_examples, class_descriptions: list):
    class_desc_str = "\n       - " + "\n       - ".join(class_descriptions)
    example_str = ""
    for example in few_shot_examples:
        example_str += "<IDEA>" + str(example['research_idea']) + "</IDEA>" + "\n"
        for i,paper in enumerate(example['related_works']):
            example_str += f"<PAPER> Paper ID [{i}]: Title: {paper['title']}. Abstract: {paper['abstract']} </PAPER>" + "\n"

        example_str += f"<REVIEW> {example['novelty_reasoning']} </REVIEW>"
        example_str += "\n\n"
    
    relevant_papers = [{
                "role": "user",
                "content": "",
            }]
    for i,d in enumerate(similar_documents):
        relevant_papers[0]['content'] += f"<PAPER> Paper ID [{i}]: Title: {d['title']}. Abstract: {d['abstract']} </PAPER>/n"

    relevant_papers[0]['content'] += """First, before you do anything else, think and reason step-by-step via chain-of-thought! Then generate the review JSON."""

    prompt = [
        {
            "role": "system",
            "content": "You are ReviewerGPT, an intelligent assistant that helps researchers evaluate the novelty of their ideas.",
        },
        {
            "role": "user",
            "content": f"""You are given some papers similar to the proposed idea (<IDEA> and </IDEA>). Your task is to evaluate the idea's novelty using the related papers (<PAPER> and </PAPER>) only.

                Types of novelty categories:
                - Not Novel: The idea closely replicates existing work with minimal or no new contributions.
                - Novel:
                    - The idea introduces new concepts or approaches that are not common in existing literature.
                    - The idea uniquely combines concepts from existing papers, but this combination does not occur in any related papers.
                    - A new application with same approach is also novel.

                Instructions:
                - Use the example reviews below to write a review for the provided idea by comparing it to the related papers.
                - Don't assume any prior knowledge about the idea.
                - When referencing a related paper, then use paper id in the review, mention it in this format: [5]. The paper ID is present between Paper ID [<paper_id>]: Title.
                - After reviewing, classify the idea into one of this category: {class_desc_str}
                - Make sure the generated review follows the format in example reviews provided below.
                - The review should be concise - around 60 to 100 words.
                - Think step-by-step before generating the final novelty score and review!


                {example_str}



                Output Format:
                ```json
                {{
                  "Class": <1|2|3|4|5>",
                  "Review": <The idea novelty is ... because...>
                }}
                ```
                """,
        },
        {"role": "assistant", "content": "Sure, please provide the IDEA."},
        {"role": "user", "content": f"Here is the idea: <IDEA> {idea} </IDEA>"},
        {"role": "assistant", "content": "Okay, now provide the related papers."},
    ]

    prompt.extend(relevant_papers)

    return prompt

def gpt_oss_inference(model, tokenizer, harmony_encoder, messages, max_new_tokens=1000, temperature=1.0, reasoning_effort="medium"):

    # ---------------------------------------------------
    # 1. Prepare Harmony-formatted inputs via chat template
    # ---------------------------------------------------
    inputs = tokenizer.apply_chat_template(
    	messages,
    	add_generation_prompt=True,
    	tokenize=True,
    	return_dict=True,
        padding=True,
        truncation=False,
    	return_tensors="pt",
        reasoning_effort=reasoning_effort,
    ).to(model.device)

    # ---------------------------------------------------
    # 2. Run inference with hidden states and attention
    # ---------------------------------------------------
    # Run generation with hidden-state capture
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,     # <-- necessary
            output_hidden_states=False,        # <-- return hidden states
            output_attentions=False,          # <-- return attention
            temperature=temperature
        )
    
    # ---------------------------------------------------
    # 3. Move inputs, output tokens and hidden states to CPU
    # ---------------------------------------------------
    inputs.to('cpu')
    outputs.sequences.to('cpu')
    torch.cuda.empty_cache()  # optional to clear cache

    # ---------------------------------------------------
    # 4. Extract completion token IDs
    # ---------------------------------------------------
    prompt_len = inputs["input_ids"].shape[-1]
    output_tokens = outputs.sequences[0][prompt_len:]

    # ---------------------------------------------------
    # 5. Harmony parsing
    # ---------------------------------------------------
    parsed_outputs = harmony_encoder.parse_messages_from_completion_tokens(output_tokens, role=Role.ASSISTANT)
    
    return parsed_outputs


#test generation
i = 0
messages = built_novelty_cot_prompt(
            idea=ds['train']['research_idea'][i],
            similar_documents=ds['train']['related_works'][i],
            few_shot_examples=few_shot_examples,
            class_descriptions=label_descriptions
        )

if model_id in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    print("Using GPT-OSS inference...")
    print(gpt_oss_inference(model=model, tokenizer=tokenizer, harmony_encoder=harmony_encoder, messages=messages, max_new_tokens=5000, temperature=1.0, reasoning_effort='low'))

else:
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map='auto')
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
        
        
        messages = built_novelty_cot_prompt(
            idea=ds['train']['research_idea'][i],
            similar_documents=ds['train']['related_works'][i][:30],
            few_shot_examples=few_shot_examples,
            class_descriptions=label_descriptions
        )
        if model_id in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            parsed_outputs = gpt_oss_inference(model=model, tokenizer=tokenizer, harmony_encoder=harmony_encoder, messages=messages, max_new_tokens=5000, temperature=1.0, reasoning_effort='low')

        else:
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map='auto')
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
run_split("test", ds["test"], "cot")