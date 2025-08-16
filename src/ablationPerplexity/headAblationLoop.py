# %%
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import random
import sys
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import perplexity

def load_results_dict(json_file_path):
    """
    Load results dictionary from JSON file.
    Returns a dictionary mapping (layer, head) string keys to perplexity values.
    """
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            return json.load(f)
    return {}

def get_next_experiment_pair(num_layers, num_heads, results_dict):
    """
    Randomly select the next (layer, head) pair that hasn't been tested yet.
    Returns None if all pairs have been tested.
    """
    all_pairs = product(range(num_layers), range(num_heads))
    untested_pairs = [pair for pair in all_pairs if f"{pair[0]},{pair[1]}" not in results_dict]
    
    if not untested_pairs:
        return None
    
    return random.choice(untested_pairs)

def save_results_dict(json_file_path, results_dict):
    """
    Save the entire results dictionary to JSON file.
    """
    with open(json_file_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

# %%
# Configuration
JSON_FILE_PATH = "head_ablation_results.json"
NUM_LAYERS = 48  # Adjust based on your model
NUM_HEADS = 64   # Adjust based on your model

# Load existing results
results = load_results_dict(JSON_FILE_PATH)
print(f"Found {len(results)} completed experiments")

# Get next experiment to run
next_pair = get_next_experiment_pair(NUM_LAYERS, NUM_HEADS, results)
if next_pair is None:
    print("All experiments completed!")
else:
    print(f"Next experiment: Layer {next_pair[0]}, Head {next_pair[1]}")

# %%
input_texts = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[
    "text"
]
input_texts = [s for s in input_texts if s != ""][:50]

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
# %%
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b").to("cuda")
# %%
results = perplexity.Perplexity().compute(
    model=model,
    tokenizer=tokenizer,
    predictions=input_texts,
)
# %%
print(results)





