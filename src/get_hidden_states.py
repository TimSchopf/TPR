#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
)
import argparse

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

# check available GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# load data
ds = load_dataset(".../RINoBench")
print(ds)
labels = load_dataset(".../RINoBench", "class_descriptions")
label_descriptions = [str(l['label'])+": "+l['description'] for l in labels['class_descriptions']]
print("Label descriptions:", label_descriptions)


# get few-shot examples
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

# harmony encoder
harmony_encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, dtype=dtype, trust_remote_code=True, device_map="auto")


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


def built_novelty_nle_prompt(idea, similar_documents, few_shot_examples, class_descriptions: list):

    class_descriptions = [descr.split(": ")[1] for descr in class_descriptions]
    class_desc_str = "\n       - " + "\n       - ".join(class_descriptions)
    example_str = ""
    for example in few_shot_examples:
        example_str += "<IDEA>" + str(example['research_idea']) + "</IDEA>" + "\n"
        for i,paper in enumerate(example['related_works']):
            example_str += f"<PAPER> Paper ID [{i}]: Title: {paper['title']}. Abstract: {paper['abstract']} </PAPER>" + "\n"

        example_str += f"<REVIEW> {example['novelty_reasoning']} </REVIEW>"
        example_str += "\n\n"
    relevant_papers = []
    for i,d in enumerate(similar_documents):
        relevant_papers.append(
            {
                "role": "user",
                "content": f"<PAPER> Paper ID [{i}]: Title: {d['title']}. Abstract: {d['abstract']} </PAPER>",
            }
        )

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
                - For reviewing, consider the following novelty categories: {class_desc_str}
                - Make sure the generated review follows the format in example reviews provided below.
                - The review should be concise - around 60 to 100 words.



                {example_str}

                Output Format:
                <REVIEW> concise review </REVIEW>
                """,
        },
        {"role": "assistant", "content": "Sure, please provide the IDEA."},
        {"role": "user", "content": f"Here is the idea: <IDEA> {idea} </IDEA>"},
        {"role": "assistant", "content": "Okay, now provide the related papers."},
    ]

    prompt.extend(relevant_papers)

    return prompt

# -----------------------------------------------------
#     Qwen3 INFERENCE (LAST-LAYER HIDDEN STATES)
# -----------------------------------------------------
def qwen_inference(model, tokenizer, messages, max_new_tokens=5000, temperature=1.0, enable_thinking=True):

    # -------------------------
    # 1) tokenize
    # -------------------------
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )

    # -------------------------
    # 2) move inputs to a GPU (first CUDA device is enough)
    # pipeline-parallel model will handle sending activations to correct GPUs
    # -------------------------
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #for k in inputs:
    #    inputs[k] = inputs[k].to(device)

    # -------------------------
    # 3) generate with hidden states
    # -------------------------
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=False,
            temperature=temperature
        )

    # -------------------------
    # 4) extract output tokens/IDs
    # -------------------------
    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = outputs.sequences[0][prompt_len:].to("cpu")

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        think_token_index = output_ids.tolist().index(151668)
    except ValueError:
        think_token_index = 0

    thinking_content = tokenizer.decode(output_ids[:think_token_index+1], skip_special_tokens=False)
    content = tokenizer.decode(output_ids[think_token_index+1:], skip_special_tokens=False)

    parsed_outputs = {"think_tokens": thinking_content, "response_tokens": content}

    # -------------------------
    # 5) collect last-layer hidden states safely
    # -------------------------
    last_layer_states = []
    # outputs.hidden_states is a tuple of (num_steps, num_layers, hidden_size)
    # for pipeline-parallel models, each element is already on correct device
    for timestep in outputs.hidden_states:
        last_layer = timestep[-1]  # take only last layer
        last_layer_states.append(last_layer.cpu())

    # -------------------------
    # 6) cleanup
    # -------------------------
    del outputs
    torch.cuda.empty_cache()

    return parsed_outputs, last_layer_states

i = 0
messages = built_novelty_nle_prompt(
            idea=ds['test']['research_idea'][i],
            similar_documents=ds['test']['related_works'][i],
            few_shot_examples=few_shot_examples,
            class_descriptions=label_descriptions
        )

qwen_inference(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=5000,
            temperature=1.0,
            enable_thinking=True
        )

# -----------------------------------------------------
#     GPT-OSS INFERENCE (LAST-LAYER HIDDEN STATES)
# -----------------------------------------------------
def gpt_oss_inference(model, tokenizer, harmony_encoder, messages, max_new_tokens=1200, temperature=1.0, reasoning_effort="low"):

    # -------------------------
    # 1) tokenize
    # -------------------------
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
        reasoning_effort=reasoning_effort,
    )

    # -------------------------
    # 2) move inputs to a GPU (first CUDA device is enough)
    # pipeline-parallel model will handle sending activations to correct GPUs
    # -------------------------
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #for k in inputs:
    #    inputs[k] = inputs[k].to(device)

    # -------------------------
    # 3) generate with hidden states
    # -------------------------
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=False,
            temperature=temperature
        )

    # -------------------------
    # 4) extract output tokens
    # -------------------------
    prompt_len = inputs["input_ids"].shape[-1]
    output_tokens = outputs.sequences[0][prompt_len:].to("cpu")

    parsed_outputs = harmony_encoder.parse_messages_from_completion_tokens(
        output_tokens, role=Role.ASSISTANT
    )

    # -------------------------
    # 5) collect last-layer hidden states safely
    # -------------------------
    last_layer_states = []
    # outputs.hidden_states is a tuple of (num_steps, num_layers, hidden_size)
    # for pipeline-parallel models, each element is already on correct device
    for timestep in outputs.hidden_states:
        last_layer = timestep[-1]  # take only last layer
        last_layer_states.append(last_layer.cpu())

    # -------------------------
    # 6) cleanup
    # -------------------------
    del outputs
    torch.cuda.empty_cache()

    return parsed_outputs, last_layer_states

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

        messages = built_novelty_nle_prompt(
            idea=data_split['research_idea'][i],
            similar_documents=data_split['related_works'][i],
            few_shot_examples=few_shot_examples,
            class_descriptions=label_descriptions
        )

        # optional message pruning to avoid OOM error due to huge prompt
        #if len(messages) > 35:
        #    messages = messages[:35]

        if "Qwen" in model.model.name_or_path:
            parsed_outputs, last_layer_states = qwen_inference(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=5000,
                temperature=1.0,
                enable_thinking=True
            )

        elif "gpt-oss" in model.model.name_or_path:
            parsed_outputs, last_layer_states = gpt_oss_inference(
                model=model,
                tokenizer=tokenizer,
                harmony_encoder=harmony_encoder,
                messages=messages,
                max_new_tokens=5000,
                temperature=1.0,
                reasoning_effort="low"
            )

        outfile = output_dir / f"{approach_name}_{split_name}_{model_id.replace('/', '-')}_{i}.pt"
        torch.save(
            {
                "parsed_outputs": parsed_outputs,
                "hidden_states_last_layer": last_layer_states
            },
            outfile
        )

        del parsed_outputs
        del last_layer_states
        torch.cuda.empty_cache()

# -----------------------------------------------------
#                     RUN
# -----------------------------------------------------
run_split("train", ds["train"],"nle_only")
run_split("test", ds["test"], "nle_only")