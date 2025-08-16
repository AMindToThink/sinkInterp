# %%
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import random
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import perplexity, ablate


def load_results_dict(json_file_path):
    """
    Load results dictionary from JSON file.
    Returns a multi-level dictionary with structure: {layer: {head_index: perplexity_values}}.
    """
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            return json.load(f)
    return {}


def get_untested_pairs(num_layers, num_heads, results_dict):
    """
    Generator that yields untested (layer, head) pairs.
    Yields pairs in order or can be consumed as needed.
    """
    for layer in range(num_layers):
        for head in range(num_heads):
            # Check if this layer-head combination has been tested
            if (
                str(layer) not in results_dict
                or str(head) not in results_dict[str(layer)]
            ):
                yield (layer, head)


def save_results_dict(json_file_path, results_dict):
    """
    Save the entire results dictionary to JSON file.
    """
    with open(json_file_path, "w") as f:
        json.dump(results_dict, f, indent=2)


# Load existing results
results = load_results_dict(JSON_FILE_PATH)
# Get list of all untested pairs and shuffle them
untested_pairs = list(get_untested_pairs(NUM_LAYERS, NUM_HEADS, results))
random.shuffle(untested_pairs)
print(f"Found {len(untested_pairs)} completed experiments")
# %%
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b").to("cuda")
# %%
# Configuration
JSON_FILE_PATH = "head_ablation_results.json"
NUM_LAYERS = len(model.model.layers)  # Adjust based on your model
assert NUM_LAYERS == 24
assert model.model.layers[0].self_attn._parameters["sinks"].shape == torch.Size([64])
NUM_HEADS = (
    model.model.layers[0].self_attn._parameters["sinks"].shape[0]
)  # Adjust based on your model
assert NUM_HEADS == 64


# %%
input_texts = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[
    "text"
]
input_texts = [s for s in input_texts if s != ""][:50]

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
# %%


if not untested_pairs:
    print("All experiments completed!")
else:
    print(f"Found {len(untested_pairs)} untested pairs to process")

    # Loop through random untested pairs
    for i, (layer, head) in enumerate(untested_pairs):
        print(f"Experiment {i+1}/{len(untested_pairs)}: Layer {layer}, Head {head}")

        with ablate.ablate_head(model, layer, head):
            ppl_results = perplexity.Perplexity().compute(
                model=model, tokenizer=tokenizer, predictions=input_texts, batch_size=64
            )

        # Store results in multi-level dictionary
        layer_str = str(layer)
        head_str = str(head)

        if layer_str not in results:
            results[layer_str] = {}

        results[layer_str][head_str] = {
            "sink": ablate.get_sink(model=model, layer=layer, head=head).item(),
            "perplexities": ppl_results,
        }

        # Save results after each experiment
        save_results_dict(JSON_FILE_PATH, results)

        print(f"  Mean perplexity: {ppl_results['mean_perplexity']:.4f}")
        print(f"  Results saved to {JSON_FILE_PATH}")

    print("All experiments completed!")
