import os
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report, f1_score, mean_absolute_error
from transformers import AutoTokenizer
import argparse

# -----------------------------------------------------
#                Command Line ARGUMENTS
# -----------------------------------------------------
def str_to_bool(value):
    if value.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif value.lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {value}')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    required=True,          # must be provided
    help="Hugging Face model ID (str)"
)

parser.add_argument(
    "--use_reasoning_token",
    type=str_to_bool,
    required=True,          # must be provided
    help="Whether to use reasoning token embeddings instead of response embeddings (bool)"
)

parser.add_argument(
    "--position_percent",
    type=float,
    required=True,          # must be provided
    help="Percentage (0.0 to 1.0) of the token position to extract embeddings from (float)"
)

args = parser.parse_args()

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
DATA_DIR = "../data/pt/"
cache_dir="../models/"
EMBED_KEY = "hidden_states_last_layer"
os.environ["HF_TOKEN"] = "..."


# --------------------------------------------------------
# HELPERS TO LOAD EMBEDDINGS AT SPECIFIC LOCATIONS
# --------------------------------------------------------
def get_target_index(length, percent):
    """Helper to calculate zero-based index from percentage (0.0 to 1.0)."""
    if length <= 0:
        return 0
    percent = max(0.0, min(1.0, percent))
    return int(percent * (length - 1))

def get_reasoning_length(data, tokenizer, model_name):
    """
    Helper to calculate the number of tokens in the reasoning/thinking phase 
    based on the model type and data structure.
    """
    reasoning_tokens = 0
    
    # Logic extracted from your original snippet
    if model_name in ["openai-gpt-oss-20b", "openai-gpt-oss-120b"]:
        text_content = data['parsed_outputs'][0].content[0].text
        # Construct the specific prompt structure for counting
        full_text = "<|channel|>analysis<|message|>" + text_content + "<|end|>"
        reasoning_tokens = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False
        )['input_ids'].shape[1]

    elif model_name in ["google-gemma-3-27b-it", "google-gemma-3-12b-it", "google-gemma-3-4b-it", "google-gemma-3-1b-it", "meta-llama-Llama-3.1-8B-Instruct", "meta-llama-Llama-3.1-70B-Instruct"]:
        # Split by review tag
        text_segment = data['parsed_outputs'].split("<REVIEW>")[0]
        reasoning_tokens = tokenizer(
            text_segment,
            return_tensors="pt",
            add_special_tokens=False
        )['input_ids'].shape[1]

    elif model_name in ["Qwen-Qwen3-32B", "Qwen-Qwen3-14B", "Qwen-Qwen3-8B", "Qwen-Qwen3-4B"]:
        text_segment = data['parsed_outputs']['think_tokens']
        reasoning_tokens = tokenizer(
            text_segment,
            return_tensors="pt",
            add_special_tokens=False
        )['input_ids'].shape[1]
        
    return reasoning_tokens

def load_response_token_embedding(path, tokenizer, model_name, position_percent=1.0):
    """
    Load embedding at X% of the *actual response* (excluding reasoning).
    0.0 = First token AFTER reasoning.
    1.0 = Last token of the file.
    """
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)

        if EMBED_KEY not in data:
            print(f"Warning: {path} missing key '{EMBED_KEY}'. Skipped.")
            return None

        hs_list = data[EMBED_KEY]
        if not isinstance(hs_list, list) or len(hs_list) == 0:
            print(f"Warning: {path} has empty hidden-state list. Skipped.")
            return None

        # 1. Calculate the length of the reasoning prefix
        reasoning_len = get_reasoning_length(data, tokenizer, model_name)
        
        # Safety: Reasoning length cannot exceed total length
        if reasoning_len > len(hs_list):
            reasoning_len = len(hs_list)

        # 2. Define the 'Response' segment
        # The response starts at index `reasoning_len` and goes to the end
        response_len = len(hs_list) - reasoning_len

        if response_len <= 0:
            print(f"Warning: {path} has no response tokens (Reasoning len {reasoning_len} >= Total {len(hs_list)}). Skipped.")
            return None

        # 3. Calculate index relative to the response segment
        relative_idx = get_target_index(response_len, position_percent)
        
        # 4. Shift index by the reasoning length
        final_idx = reasoning_len + relative_idx
        
        target_h = hs_list[final_idx]
        target_h = target_h.squeeze(0).squeeze(0)

        del data
        del hs_list
        return target_h

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_reasoning_token_embedding(path, tokenizer, model_name, position_percent=1.0):
    """
    Load embedding at X% of the *reasoning sequence*.
    0.0 = First reasoning token.
    1.0 = Last reasoning token.
    """
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)

        if EMBED_KEY not in data:
            print(f"Warning: {path} missing key '{EMBED_KEY}'. Skipped.")
            return None

        hs_list = data[EMBED_KEY]
        if not isinstance(hs_list, list) or len(hs_list) == 0:
            print(f"Warning: {path} has empty hidden-state list. Skipped.")
            return None

        # 1. Get reasoning length
        reasoning_len = get_reasoning_length(data, tokenizer, model_name)

        # Safety:
        if reasoning_len > len(hs_list):
            print(f"Warning: Reasoning token count {reasoning_len} exceeds hidden states {len(hs_list)}. Using max available.")
            reasoning_len = len(hs_list)
        
        if reasoning_len == 0:
            print(f"Warning: {path} has 0 reasoning tokens. Skipped.")
            return None

        # 2. Calculate index within the reasoning segment (0 to reasoning_len - 1)
        target_idx = get_target_index(reasoning_len, position_percent)

        # 3. if index is 0, ensure we get the first reasoning token by setting it + 1 (since index 0 returns embeddings of all input tokens before reasoning/generation starts)
        if target_idx == 0:
            target_idx+=1
        
        target_reasoning_h = hs_list[target_idx]
        target_reasoning_h = target_reasoning_h.squeeze(0).squeeze(0)

        del data
        del hs_list
        return target_reasoning_h

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_all_embeddings(approach_name: str, model_name: str, tokenizer, use_reasoning_token=False, position_percent=1.0):
    print(f"Loading train + test embeddings from: {DATA_DIR} for model: {model_name} and approach: {approach_name} at position {position_percent*100}% (using reasoning token: {use_reasoning_token})")

    all_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".pt")
    ])

    # Regex patterns
    train_re = re.compile(approach_name+"_train_"+model_name+r"_\d+\.pt")
    test_re  = re.compile(approach_name+"_test_"+model_name+r"_\d+\.pt")

    x_train = []
    x_test = []

    for fname in tqdm(all_files, desc="Processing .pt files"):
        path = os.path.join(DATA_DIR, fname)
        emb = None

        if train_re.match(fname) or test_re.match(fname):
            if use_reasoning_token:
                # Load X% of reasoning
                emb = load_reasoning_token_embedding(path, tokenizer, model_name, position_percent=position_percent)
            else:
                # Load X% of response (excluding reasoning)
                emb = load_response_token_embedding(path, tokenizer, model_name, position_percent=position_percent)
            
            if emb is not None:
                if train_re.match(fname):
                    x_train.append(emb)
                else:
                    x_test.append(emb)

    return x_train, x_test

# -----------------------------------------------------
#      LOAD TRAIN/TEST EMBEDDINGS (x data)
# -----------------------------------------------------
model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, device_map='cpu')
x_train, x_test = load_all_embeddings(approach_name="nle_only_r2", model_name=model_id.replace("/","-"), tokenizer=tokenizer, use_reasoning_token=args.use_reasoning_token, position_percent=args.position_percent)
print(len(x_train), "train embeddings loaded.")
print(len(x_test), "test embeddings loaded.")

# load y data
ds = load_dataset("TimSchopf/RINoBench")
labels = load_dataset("TimSchopf/RINoBench", "class_descriptions")
label_descriptions = [str(l['label'])+": "+l['description'] for l in labels['class_descriptions']]

y_train = list(ds["train"]["novelty_score"])
y_test = list(ds["test"]["novelty_score"])

print("converting data to numpy arrays...")
#convert to compatible values for sklearn
x_train = np.stack([t.to(torch.float32).cpu().numpy() for t in x_train])
x_test = np.stack([t.to(torch.float32).cpu().numpy() for t in x_test])
y_train = np.array(y_train)
y_test = np.array(y_test)

# train logistic regression probing classifier
print("Training probing classifier...")
clf = LogisticRegression(
    max_iter=100,
    solver="lbfgs"
).fit(x_train, y_train)

# predict on test set using probing classifier
y_pred = clf.predict(x_test)

# -----------------------------------------------------------
# Evaluation scores of probing classifier predictions
# -----------------------------------------------------------
print(f"Evaluation results for model: {model_id}, use_reasoning_token: {args.use_reasoning_token}, position_percent: {args.position_percent}")
macro_f1 = f1_score(y_test, y_pred, average="macro")
print("Macro F1 Score:", macro_f1)

# Per-class F1 (classification_report includes this)
print("\nPer-Class F1 Scores:")
print(classification_report(y_test, y_pred, digits=4))

# Mean Absolute Error
print("\nMean Absolute Error:")
print(float(mean_absolute_error(y_test, y_pred)))