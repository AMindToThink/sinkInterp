# %%
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM
import torch
from utils.ablate import get_sink
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import scipy.stats as stats

# %%
# Load the model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")

# %%
# Load the matrix norms data
print("Loading matrix norms data...")
with open("head_norms.json", "r") as f:
    head_norms = json.load(f)

# %%
# Configuration
NUM_LAYERS = len(model.model.layers)
NUM_HEADS = model.model.layers[0].self_attn.sinks.shape[0]

print(f"Model has {NUM_LAYERS} layers and {NUM_HEADS} heads per layer")

# Extract sink values and matrix norms, separating even and odd layers
even_sink_values = []
even_fro_norms = []
even_spectral_norms = []
even_expected_norms = []
even_layer_indices = []

odd_sink_values = []
odd_fro_norms = []
odd_spectral_norms = []
odd_expected_norms = []
odd_layer_indices = []

# %%
print("Extracting sink values and matrix norms for even and odd layers...")
for layer in tqdm(range(NUM_LAYERS)):
    for head in range(NUM_HEADS):
        # Get sink value using the ablate.py function
        sink_value = get_sink(model, layer, head).item()

        # Get matrix norms from the JSON data
        layer_str = str(layer)
        head_str = str(head)

        if layer_str in head_norms and head_str in head_norms[layer_str]:
            norms_data = head_norms[layer_str][head_str]

            # Separate even and odd layers
            if layer % 2 == 0:  # Even layer
                even_sink_values.append(sink_value)
                even_fro_norms.append(norms_data["fro"])
                even_spectral_norms.append(norms_data["spectral"])
                even_expected_norms.append(norms_data["expected_quadratic_form_norm"])
                even_layer_indices.append(layer)
            else:  # Odd layer
                odd_sink_values.append(sink_value)
                odd_fro_norms.append(norms_data["fro"])
                odd_spectral_norms.append(norms_data["spectral"])
                odd_expected_norms.append(norms_data["expected_quadratic_form_norm"])
                odd_layer_indices.append(layer)

# Convert to numpy arrays for easier handling
even_sink_values = np.array(even_sink_values)
even_fro_norms = np.array(even_fro_norms)
even_spectral_norms = np.array(even_spectral_norms)
even_expected_norms = np.array(even_expected_norms)

odd_sink_values = np.array(odd_sink_values)
odd_fro_norms = np.array(odd_fro_norms)
odd_spectral_norms = np.array(odd_spectral_norms)
odd_expected_norms = np.array(odd_expected_norms)

# Compute log transformations
even_log_fro_norms = np.log(even_fro_norms)
even_log_spectral_norms = np.log(even_spectral_norms)
even_log_expected_norms = np.log(even_expected_norms)

odd_log_fro_norms = np.log(odd_fro_norms)
odd_log_spectral_norms = np.log(odd_spectral_norms)
odd_log_expected_norms = np.log(odd_expected_norms)

print(
    f"Collected data for {len(even_sink_values)} even layer heads and {len(odd_sink_values)} odd layer heads"
)
print(f"\nEven layers:")
print(
    f"  Sink values range: {even_sink_values.min():.3f} to {even_sink_values.max():.3f}"
)
print(
    f"  Frobenius norms range: {even_fro_norms.min():.3f} to {even_fro_norms.max():.3f}"
)
print(
    f"  Expected norms range: {even_expected_norms.min():.3f} to {even_expected_norms.max():.3f}"
)

print(f"\nOdd layers:")
print(
    f"  Sink values range: {odd_sink_values.min():.3f} to {odd_sink_values.max():.3f}"
)
print(
    f"  Frobenius norms range: {odd_fro_norms.min():.3f} to {odd_fro_norms.max():.3f}"
)
print(
    f"  Expected norms range: {odd_expected_norms.min():.3f} to {odd_expected_norms.max():.3f}"
)

# %%
# Create plots comparing even and odd layers
fig, axes = plt.subplots(3, 3, figsize=(18, 18))

# Plot 1: Sink values vs Frobenius norm (Even vs Odd)
axes[0, 0].scatter(
    even_sink_values, even_fro_norms, alpha=0.6, s=20, color="blue", label="Even layers"
)
axes[0, 0].scatter(
    odd_sink_values, odd_fro_norms, alpha=0.6, s=20, color="red", label="Odd layers"
)
axes[0, 0].set_xlabel("Sink Value")
axes[0, 0].set_ylabel("Frobenius Norm")
axes[0, 0].set_title("Sink Value vs Frobenius Norm (Even vs Odd Layers)")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Sink values vs Spectral norm (Even vs Odd)
axes[0, 1].scatter(
    even_sink_values,
    even_spectral_norms,
    alpha=0.6,
    s=20,
    color="blue",
    label="Even layers",
)
axes[0, 1].scatter(
    odd_sink_values,
    odd_spectral_norms,
    alpha=0.6,
    s=20,
    color="red",
    label="Odd layers",
)
axes[0, 1].set_xlabel("Sink Value")
axes[0, 1].set_ylabel("Spectral Norm")
axes[0, 1].set_title("Sink Value vs Spectral Norm (Even vs Odd Layers)")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Sink values vs Expected norm (Even vs Odd)
axes[0, 2].scatter(
    even_sink_values,
    even_expected_norms,
    alpha=0.6,
    s=20,
    color="blue",
    label="Even layers",
)
axes[0, 2].scatter(
    odd_sink_values,
    odd_expected_norms,
    alpha=0.6,
    s=20,
    color="red",
    label="Odd layers",
)
axes[0, 2].set_xlabel("Sink Value")
axes[0, 2].set_ylabel("Expected Quadratic Form Norm")
axes[0, 2].set_title("Sink Value vs Expected Norm (Even vs Odd Layers)")
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# Plot 4: Log Sink values vs Log Frobenius norm (Even vs Odd)
axes[1, 0].scatter(
    even_sink_values,
    even_log_fro_norms,
    alpha=0.6,
    s=20,
    color="blue",
    label="Even layers",
)
axes[1, 0].scatter(
    odd_sink_values, odd_log_fro_norms, alpha=0.6, s=20, color="red", label="Odd layers"
)
axes[1, 0].set_xlabel("Sink Value")
axes[1, 0].set_ylabel("Log Frobenius Norm")
axes[1, 0].set_title("Sink Value vs Log Frobenius Norm (Even vs Odd Layers)")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 5: Log Sink values vs Log Spectral norm (Even vs Odd)
axes[1, 1].scatter(
    even_sink_values,
    even_log_spectral_norms,
    alpha=0.6,
    s=20,
    color="blue",
    label="Even layers",
)
axes[1, 1].scatter(
    odd_sink_values,
    odd_log_spectral_norms,
    alpha=0.6,
    s=20,
    color="red",
    label="Odd layers",
)
axes[1, 1].set_xlabel("Sink Value")
axes[1, 1].set_ylabel("Log Spectral Norm")
axes[1, 1].set_title("Sink Value vs Log Spectral Norm (Even vs Odd Layers)")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# Plot 6: Log Sink values vs Log Expected norm (Even vs Odd)
axes[1, 2].scatter(
    even_sink_values,
    even_log_expected_norms,
    alpha=0.6,
    s=20,
    color="blue",
    label="Even layers",
)
axes[1, 2].scatter(
    odd_sink_values,
    odd_log_expected_norms,
    alpha=0.6,
    s=20,
    color="red",
    label="Odd layers",
)
axes[1, 2].set_xlabel("Sink Value")
axes[1, 2].set_ylabel("Log Expected Quadratic Form Norm")
axes[1, 2].set_title("Sink Value vs Log Expected Norm (Even vs Odd Layers)")
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

# Plot 7: Even layers only - all norms
scaler_even = StandardScaler()
even_fro_norm_scaled = scaler_even.fit_transform(
    even_fro_norms.reshape(-1, 1)
).flatten()
even_spectral_norm_scaled = scaler_even.fit_transform(
    even_spectral_norms.reshape(-1, 1)
).flatten()
even_expected_norm_scaled = scaler_even.fit_transform(
    even_expected_norms.reshape(-1, 1)
).flatten()

axes[2, 0].scatter(
    even_sink_values,
    even_fro_norm_scaled,
    alpha=0.4,
    s=15,
    label="Frobenius",
    color="darkblue",
)
axes[2, 0].scatter(
    even_sink_values,
    even_spectral_norm_scaled,
    alpha=0.4,
    s=15,
    label="Spectral",
    color="lightblue",
)
axes[2, 0].scatter(
    even_sink_values,
    even_expected_norm_scaled,
    alpha=0.4,
    s=15,
    label="Expected",
    color="navy",
)
axes[2, 0].set_xlabel("Sink Value")
axes[2, 0].set_ylabel("Normalized Matrix Norm")
axes[2, 0].set_title("Even Layers: Sink Value vs All Matrix Norms (Normalized)")
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

# Plot 8: Odd layers only - all norms
scaler_odd = StandardScaler()
odd_fro_norm_scaled = scaler_odd.fit_transform(odd_fro_norms.reshape(-1, 1)).flatten()
odd_spectral_norm_scaled = scaler_odd.fit_transform(
    odd_spectral_norms.reshape(-1, 1)
).flatten()
odd_expected_norm_scaled = scaler_odd.fit_transform(
    odd_expected_norms.reshape(-1, 1)
).flatten()

axes[2, 1].scatter(
    odd_sink_values,
    odd_fro_norm_scaled,
    alpha=0.4,
    s=15,
    label="Frobenius",
    color="darkred",
)
axes[2, 1].scatter(
    odd_sink_values,
    odd_spectral_norm_scaled,
    alpha=0.4,
    s=15,
    label="Spectral",
    color="orange",
)
axes[2, 1].scatter(
    odd_sink_values,
    odd_expected_norm_scaled,
    alpha=0.4,
    s=15,
    label="Expected",
    color="maroon",
)
axes[2, 1].set_xlabel("Sink Value")
axes[2, 1].set_ylabel("Normalized Matrix Norm")
axes[2, 1].set_title("Odd Layers: Sink Value vs All Matrix Norms (Normalized)")
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend()

# Plot 9: Distribution comparison
axes[2, 2].hist(
    even_sink_values,
    bins=30,
    alpha=0.5,
    label="Even layers",
    color="blue",
    density=True,
)
axes[2, 2].hist(
    odd_sink_values, bins=30, alpha=0.5, label="Odd layers", color="red", density=True
)
axes[2, 2].set_xlabel("Sink Value")
axes[2, 2].set_ylabel("Density")
axes[2, 2].set_title("Distribution of Sink Values (Even vs Odd Layers)")
axes[2, 2].grid(True, alpha=0.3)
axes[2, 2].legend()

plt.tight_layout()
plt.savefig("even_odd_sink_vs_matrix_norms.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Calculate correlations for even layers
even_correlation_fro = np.corrcoef(even_sink_values, even_fro_norms)[0, 1]
even_correlation_spectral = np.corrcoef(even_sink_values, even_spectral_norms)[0, 1]
even_correlation_expected = np.corrcoef(even_sink_values, even_expected_norms)[0, 1]
even_correlation_log_fro = np.corrcoef(even_sink_values, even_log_fro_norms)[0, 1]
even_correlation_log_spectral = np.corrcoef(even_sink_values, even_log_spectral_norms)[
    0, 1
]
even_correlation_log_expected = np.corrcoef(even_sink_values, even_log_expected_norms)[
    0, 1
]

# Calculate correlations for odd layers
odd_correlation_fro = np.corrcoef(odd_sink_values, odd_fro_norms)[0, 1]
odd_correlation_spectral = np.corrcoef(odd_sink_values, odd_spectral_norms)[0, 1]
odd_correlation_expected = np.corrcoef(odd_sink_values, odd_expected_norms)[0, 1]
odd_correlation_log_fro = np.corrcoef(odd_sink_values, odd_log_fro_norms)[0, 1]
odd_correlation_log_spectral = np.corrcoef(odd_sink_values, odd_log_spectral_norms)[
    0, 1
]
odd_correlation_log_expected = np.corrcoef(odd_sink_values, odd_log_expected_norms)[
    0, 1
]

print(f"\n=== CORRELATIONS CONTROLLING FOR LAYER EVENNESS ===")
print(f"\nEven Layers (n={len(even_sink_values)}):")
print(f"  Sink vs Frobenius norm: {even_correlation_fro:.4f}")
print(f"  Sink vs Spectral norm: {even_correlation_spectral:.4f}")
print(f"  Sink vs Expected norm: {even_correlation_expected:.4f}")
print(f"  Sink vs Log Frobenius norm: {even_correlation_log_fro:.4f}")
print(f"  Sink vs Log Spectral norm: {even_correlation_log_spectral:.4f}")
print(f"  Sink vs Log Expected norm: {even_correlation_log_expected:.4f}")

print(f"\nOdd Layers (n={len(odd_sink_values)}):")
print(f"  Sink vs Frobenius norm: {odd_correlation_fro:.4f}")
print(f"  Sink vs Spectral norm: {odd_correlation_spectral:.4f}")
print(f"  Sink vs Expected norm: {odd_correlation_expected:.4f}")
print(f"  Sink vs Log Frobenius norm: {odd_correlation_log_fro:.4f}")
print(f"  Sink vs Log Spectral norm: {odd_correlation_log_spectral:.4f}")
print(f"  Sink vs Log Expected norm: {odd_correlation_log_expected:.4f}")


# %%
# Statistical significance tests for even and odd layers separately
def correlation_test(x, y, layer_type, norm_type):
    """Perform correlation test with statistical significance"""
    correlation_coeff, p_value_two_tailed = pearsonr(x, y)

    n = len(x)
    t_statistic = correlation_coeff * np.sqrt((n - 2) / (1 - correlation_coeff**2))
    df = n - 2

    # One-tailed p-value (testing for positive correlation)
    p_value_one_tailed = 1 - stats.t.cdf(t_statistic, df)

    print(f"\n--- {layer_type} Layers: Sink vs {norm_type} ---")
    print(f"Correlation coefficient: {correlation_coeff:.6f}")
    print(f"Sample size: {n}")
    print(f"T-statistic: {t_statistic:.6f}")
    print(f"One-tailed p-value (H1: r > 0): {p_value_one_tailed:.6e}")
    print(f"Two-tailed p-value: {p_value_two_tailed:.6e}")

    # Interpret results
    alpha = 0.05
    if p_value_one_tailed < alpha:
        print(f"Result: SIGNIFICANT at α = {alpha} (positive correlation)")
    else:
        print(f"Result: NOT SIGNIFICANT at α = {alpha}")

    return correlation_coeff, p_value_one_tailed, p_value_two_tailed


print(f"\n=== STATISTICAL SIGNIFICANCE TESTS ===")

# Test even layers
correlation_test(even_sink_values, even_expected_norms, "Even", "Expected Norm")
correlation_test(even_sink_values, even_fro_norms, "Even", "Frobenius Norm")
correlation_test(even_sink_values, even_spectral_norms, "Even", "Spectral Norm")

# Test odd layers
correlation_test(odd_sink_values, odd_expected_norms, "Odd", "Expected Norm")
correlation_test(odd_sink_values, odd_fro_norms, "Odd", "Frobenius Norm")
correlation_test(odd_sink_values, odd_spectral_norms, "Odd", "Spectral Norm")

# %%
# Save the data separated by even/odd layers
data_dict = {
    "even_layers": {
        "sink_values": even_sink_values.tolist(),
        "frobenius_norms": even_fro_norms.tolist(),
        "spectral_norms": even_spectral_norms.tolist(),
        "expected_norms": even_expected_norms.tolist(),
        "log_frobenius_norms": even_log_fro_norms.tolist(),
        "log_spectral_norms": even_log_spectral_norms.tolist(),
        "log_expected_norms": even_log_expected_norms.tolist(),
        "layer_indices": even_layer_indices,
        "correlations": {
            "sink_vs_frobenius": even_correlation_fro,
            "sink_vs_spectral": even_correlation_spectral,
            "sink_vs_expected": even_correlation_expected,
            "sink_vs_log_frobenius": even_correlation_log_fro,
            "sink_vs_log_spectral": even_correlation_log_spectral,
            "sink_vs_log_expected": even_correlation_log_expected,
        },
    },
    "odd_layers": {
        "sink_values": odd_sink_values.tolist(),
        "frobenius_norms": odd_fro_norms.tolist(),
        "spectral_norms": odd_spectral_norms.tolist(),
        "expected_norms": odd_expected_norms.tolist(),
        "log_frobenius_norms": odd_log_fro_norms.tolist(),
        "log_spectral_norms": odd_log_spectral_norms.tolist(),
        "log_expected_norms": odd_log_expected_norms.tolist(),
        "layer_indices": odd_layer_indices,
        "correlations": {
            "sink_vs_frobenius": odd_correlation_fro,
            "sink_vs_spectral": odd_correlation_spectral,
            "sink_vs_expected": odd_correlation_expected,
            "sink_vs_log_frobenius": odd_correlation_log_fro,
            "sink_vs_log_spectral": odd_correlation_log_spectral,
            "sink_vs_log_expected": odd_correlation_log_expected,
        },
    },
    "summary": {
        "total_even_heads": len(even_sink_values),
        "total_odd_heads": len(odd_sink_values),
        "even_layers_range": [min(even_layer_indices), max(even_layer_indices)],
        "odd_layers_range": [min(odd_layer_indices), max(odd_layer_indices)],
    },
}

with open("even_odd_sink_vs_norms_data.json", "w") as f:
    json.dump(data_dict, f, indent=2)

print(f"\n=== ANALYSIS COMPLETE ===")
print(f"Plot saved as 'even_odd_sink_vs_matrix_norms.png'")
print(f"Raw data saved as 'even_odd_sink_vs_norms_data.json'")
print(f"Total heads analyzed: {len(even_sink_values) + len(odd_sink_values)}")
print(f"Even layer heads: {len(even_sink_values)}")
print(f"Odd layer heads: {len(odd_sink_values)}")

# %%
