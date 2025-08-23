import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

layers_20b = 24
heads_per_layer = 64

# Load the JSON files
with open("head_norms.json", "r") as f:
    norms_20b = json.load(f)

with open("head_norms_120b.json", "r") as f:
    norms_120b = json.load(f)

# Extract expected_quadratic_form_norm values for layers 0-23
norms_20b_values = []
norms_120b_values = []

for layer in range(24):  # layers 0-23
    layer_str = str(layer)
    if layer_str in norms_20b and layer_str in norms_120b:
        for head_idx in range(heads_per_layer):
            head_str = str(head_idx)
            if (
                head_str in norms_20b[layer_str]
                and head_str in norms_120b[layer_str]
                and "expected_quadratic_form_norm" in norms_20b[layer_str][head_str]
                and "expected_quadratic_form_norm" in norms_120b[layer_str][head_str]
            ):

                norm_20b = norms_20b[layer_str][head_str][
                    "expected_quadratic_form_norm"
                ]
                norm_120b = norms_120b[layer_str][head_str][
                    "expected_quadratic_form_norm"
                ]

                norms_20b_values.append(norm_20b)
                norms_120b_values.append(norm_120b)

# Calculate correlation
r_value, p_value = pearsonr(norms_20b_values, norms_120b_values)

print(f"Correlation coefficient (R): {r_value:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Number of data points: {len(norms_20b_values)}")

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(norms_20b_values, norms_120b_values, alpha=0.6, s=20)
plt.xlabel("Expected Quadratic Form Norm (20B model)")
plt.ylabel("Expected Quadratic Form Norm (120B model)")
plt.title(
    f"Correlation between 20B and 120B Model Head Norms\nR = {r_value:.4f}, p = {p_value:.4e}"
)

# Add diagonal line for reference
min_val = min(min(norms_20b_values), min(norms_120b_values))
max_val = max(max(norms_20b_values), max(norms_120b_values))
plt.plot(
    [min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="y=x reference line"
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
