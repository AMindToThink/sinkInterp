# %%
from transformers import AutoModelForCausalLM
import torch
import json
from itertools import product
from utils import attention_info
from tqdm import tqdm
import sys

# Change to 120b for larger test
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
# %%
# Configuration
NUM_LAYERS = len(model.model.layers)  # Adjust based on your model
NUM_HEADS = model.model.layers[0].self_attn.sinks.shape[0]  # Adjust based on your model
# %%
results_dict = {}
for layer in range(NUM_LAYERS):
    results_dict[str(layer)] = {}

for layer, head_idx in tqdm(product(range(NUM_LAYERS), range(NUM_HEADS))):
    results_for_layer_head = {}
    head_parameters_dict = attention_info.get_head_projections(
        model=model, layer_idx=layer, head_idx=head_idx
    )
    quadratic_form = attention_info.findQuadraticForm(head_parameters_dict).float()
    results_for_layer_head["expected_quadratic_form_norm"] = (
        attention_info.expected_quadratic_form_norm(quadratic_form)
    ).item()
    results_for_layer_head["fro"] = torch.norm(quadratic_form).item()
    assert results_for_layer_head["fro"] == torch.linalg.norm(quadratic_form).item()
    results_for_layer_head["spectral"] = torch.linalg.norm(quadratic_form, ord=2).item()
    results_dict[str(layer)][str(head_idx)] = results_for_layer_head


with open("head_norms.json", "w") as f:
    json.dump(results_dict, f, indent=2)

# %%
