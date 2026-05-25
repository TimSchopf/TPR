import torch
import os
import re
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report, f1_score, mean_absolute_error

os.environ["HF_TOKEN"] = "..."

# load y data
ds = load_dataset("TimSchopf/RINoBench")
labels = load_dataset("TimSchopf/RINoBench", "class_descriptions")
label_descriptions = [str(l['label'])+": "+l['description'] for l in labels['class_descriptions']]

test_indices = list(range(0,len(ds["test"])))
train_indices = list(range(0,len(ds["train"])))
y_train = [ds["train"]["novelty_score"][idx] for idx in train_indices]
y_test = [ds["test"]["novelty_score"][idx] for idx in test_indices]

# load textual x_test data
def extract_prediction_value(text: str, approach: str) -> int:
    """
    Safely extracts the integer value of the 'Class' key from a string.
    Handles double quotes, single quotes, or **Class** notation.
    Raises ValueError if not found.
    """
    if approach in ["novelty_checker", "cot"]:
        # Regex pattern:
        # - optional ** around Class
        # - single or double quotes around Class
        # - optional spaces around colon
        # - captures integer value
        pattern = r'(?i)(?:\*\*)?\s*["\']?Class["\']?\s*(?:\*\*)?\s*[:=]\s*(?:\*\*)?\s*["\']?(\d+)'

    elif approach in ["ai_researcher", "naive_prompting", "naive_prompting_low_reasoning"]:
        pattern = r'(?:\*\*)?\s*(?:["\']?)novelty_score(?:["\']?)\s*(?:\*\*)?\s*:\s*(\d+)'

    elif approach == "ai_scientist":
        pattern = r'(?:\*\*)?\s*(?:["\']?)NOVELTY(?:["\']?)\s*(?:\*\*)?\s*:\s*(\d+)'

    elif approach == "research_agent":
        pattern = r'(?:\*\*)?\s*(?:["\']?)Rating(?: \(1[\-\u2010\u2011\u2012\u2013\u2014]5\))?(?:["\']?)\s*(?:\*\*)?\s*:\s*(?:["\']?)\s*(\d+)\s*(?:["\']?)'

    elif approach == "moose":
        pattern = r'(?:\*\*)?\s*"?Novelty score"?\s*(?:\*\*)?\s*:\s*"?(\d+)"?'

    match = re.search(pattern, text)
    if match:
        return int(re.sub(r'\D+', '', match.group(1)))
    
    raise ValueError("No pattern key with an integer value found.")
    
approach_name="cot"
model_name="openai-gpt-oss-20b"
textual_predictions = []
for i in test_indices:
    o = torch.load("../data/pt/"+approach_name+"_test_"+model_name+"_"+str(i)+".pt", map_location="cpu", weights_only=False)['parsed_outputs']
    #print(o)
    if model_name in ["openai-gpt-oss-20b","openai-gpt-oss-120b"]:
        for message in o:
            if message.channel == 'final':
                textual_predictions.append(extract_prediction_value(message.content[0].text, approach_name))
    elif model_name in ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-2.5-flash", "google-gemma-3-27b-it", "google-gemma-3-12b-it", "google-gemma-3-4b-it", "google-gemma-3-1b-it", "meta-llama-Llama-3.1-8B-Instruct", "meta-llama-Llama-3.1-70B-Instruct", "Qwen-Qwen3-4B", "Qwen-Qwen3-8B", "Qwen-Qwen3-14B", "Qwen-Qwen3-32B", "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"]:
        try:
            textual_predictions.append(extract_prediction_value(o, approach_name))
        except:
            print(i)
            textual_predictions.append(4)

print("Approach:", approach_name)
print("Model:", model_name)
print("No. textual predictions:", len(textual_predictions))

# convert to numpy arrays
y_test = np.array(y_test)
textual_predictions = np.array(textual_predictions)

# -----------------------------------------
# Evaluation scores of textual LLM predictions
# -----------------------------------------
macro_f1 = f1_score(y_test, textual_predictions, average="macro")
print("Macro F1 Score:", macro_f1)

# Per-class F1 (classification_report includes this)
print("\nPer-Class F1 Scores:\n")
print(classification_report(y_test, textual_predictions, digits=4))

# Mean Absolute Error
print("\nMean Absolute Error:\n")
print(float(mean_absolute_error(y_test, textual_predictions)))