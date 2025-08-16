import json
import os
import random
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


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


JSON_FILE_PATH = "head_ablation_results.json"

results = load_results_dict(JSON_FILE_PATH)
NUM_HEADS = 64
NUM_LAYERS = 24

# Extract layer 5 results for plotting
layer_5_sink = []
layer_5_perplexity = []

for i in range(1, 24, 2):
    for head_idx, head_data in results[str(i)].items():
        if (
            isinstance(head_data, dict)
            and "sink" in head_data
            and "mean_perplexity" in head_data["perplexities"]
        ):
            layer_5_sink.append(head_data["sink"])
            layer_5_perplexity.append(head_data["perplexities"]["mean_perplexity"])


# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(layer_5_sink, layer_5_perplexity, alpha=0.7)
plt.xlabel("Sink")
plt.ylabel("Mean Perplexity")
plt.title("Window Layers: Sink vs Mean Perplexity")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Plotted {len(layer_5_sink)} data points from layer 5")
