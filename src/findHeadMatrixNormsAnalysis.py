# %%
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM
import torch
from utils.ablate import get_sink
from tqdm import tqdm

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

# Extract sink values and matrix norms
sink_values = []
fro_norms = []
spectral_norms = []
expected_norms = []
# %%
print("Extracting sink values and matrix norms...")
for layer in tqdm(range(NUM_LAYERS)):
    for head in range(NUM_HEADS):
        # Get sink value using the ablate.py function
        sink_value = get_sink(model, layer, head).item()

        # Get matrix norms from the JSON data
        layer_str = str(layer)
        head_str = str(head)

        if layer_str in head_norms and head_str in head_norms[layer_str]:
            norms_data = head_norms[layer_str][head_str]

            sink_values.append(sink_value)
            fro_norms.append(norms_data["fro"])
            spectral_norms.append(norms_data["spectral"])
            expected_norms.append(norms_data["expected_quadratic_form_norm"])

# Convert to numpy arrays for easier handling
sink_values = np.array(sink_values)
fro_norms = np.array(fro_norms)
spectral_norms = np.array(spectral_norms)
expected_norms = np.array(expected_norms)

# Compute log transformations
log_fro_norms = np.log(fro_norms)
log_spectral_norms = np.log(spectral_norms)
log_expected_norms = np.log(expected_norms)

print(f"Collected data for {len(sink_values)} heads")
print(f"Sink values range: {sink_values.min():.3f} to {sink_values.max():.3f}")
print(f"Frobenius norms range: {fro_norms.min():.3f} to {fro_norms.max():.3f}")
print(f"Spectral norms range: {spectral_norms.min():.3f} to {spectral_norms.max():.3f}")
print(
    f"Log Frobenius norms range: {log_fro_norms.min():.3f} to {log_fro_norms.max():.3f}"
)
print(
    f"Log Spectral norms range: {log_spectral_norms.min():.3f} to {log_spectral_norms.max():.3f}"
)

# Create plots
fig, axes = plt.subplots(4, 2, figsize=(15, 24))

# Plot 1: Sink values vs Frobenius norm
axes[0, 0].scatter(sink_values, fro_norms, alpha=0.6, s=20)
axes[0, 0].set_xlabel("Sink Value")
axes[0, 0].set_ylabel("Frobenius Norm")
axes[0, 0].set_title("Sink Value vs Frobenius Norm")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Sink values vs Spectral norm
axes[0, 1].scatter(sink_values, spectral_norms, alpha=0.6, s=20, color="orange")
axes[0, 1].set_xlabel("Sink Value")
axes[0, 1].set_ylabel("Spectral Norm")
axes[0, 1].set_title("Sink Value vs Spectral Norm")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Sink values vs Expected quadratic form norm
axes[1, 0].scatter(sink_values, expected_norms, alpha=0.6, s=20, color="green")
axes[1, 0].set_xlabel("Sink Value")
axes[1, 0].set_ylabel("Expected Quadratic Form Norm")
axes[1, 0].set_title("Sink Value vs Expected Quadratic Form Norm")
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Sink values vs Log Frobenius norm
axes[1, 1].scatter(sink_values, log_fro_norms, alpha=0.6, s=20, color="purple")
axes[1, 1].set_xlabel("Sink Value")
axes[1, 1].set_ylabel("Log Frobenius Norm")
axes[1, 1].set_title("Sink Value vs Log Frobenius Norm")
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Sink values vs Log Spectral norm
axes[2, 0].scatter(sink_values, log_spectral_norms, alpha=0.6, s=20, color="red")
axes[2, 0].set_xlabel("Sink Value")
axes[2, 0].set_ylabel("Log Spectral Norm")
axes[2, 0].set_title("Sink Value vs Log Spectral Norm")
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: Sink values vs Log Expected norm
axes[2, 1].scatter(sink_values, log_expected_norms, alpha=0.6, s=20, color="brown")
axes[2, 1].set_xlabel("Sink Value")
axes[2, 1].set_ylabel("Log Expected Quadratic Form Norm")
axes[2, 1].set_title("Sink Value vs Log Expected Quadratic Form Norm")
axes[2, 1].grid(True, alpha=0.3)

# Plot 7: All norms together (normalized for comparison)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
fro_norm_scaled = scaler.fit_transform(fro_norms.reshape(-1, 1)).flatten()
spectral_norm_scaled = scaler.fit_transform(spectral_norms.reshape(-1, 1)).flatten()
expected_norm_scaled = scaler.fit_transform(expected_norms.reshape(-1, 1)).flatten()

axes[3, 0].scatter(
    sink_values,
    fro_norm_scaled,
    alpha=0.4,
    s=15,
    label="Frobenius (scaled)",
    color="blue",
)
axes[3, 0].scatter(
    sink_values,
    spectral_norm_scaled,
    alpha=0.4,
    s=15,
    label="Spectral (scaled)",
    color="orange",
)
axes[3, 0].scatter(
    sink_values,
    expected_norm_scaled,
    alpha=0.4,
    s=15,
    label="Expected (scaled)",
    color="green",
)
axes[3, 0].set_xlabel("Sink Value")
axes[3, 0].set_ylabel("Normalized Matrix Norm")
axes[3, 0].set_title("Sink Value vs All Matrix Norms (Normalized)")
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].legend()

# Plot 8: All log norms together (normalized for comparison)
log_fro_norm_scaled = scaler.fit_transform(log_fro_norms.reshape(-1, 1)).flatten()
log_spectral_norm_scaled = scaler.fit_transform(
    log_spectral_norms.reshape(-1, 1)
).flatten()
log_expected_norm_scaled = scaler.fit_transform(
    log_expected_norms.reshape(-1, 1)
).flatten()

axes[3, 1].scatter(
    sink_values,
    log_fro_norm_scaled,
    alpha=0.4,
    s=15,
    label="Log Frobenius (scaled)",
    color="purple",
)
axes[3, 1].scatter(
    sink_values,
    log_spectral_norm_scaled,
    alpha=0.4,
    s=15,
    label="Log Spectral (scaled)",
    color="red",
)
axes[3, 1].scatter(
    sink_values,
    log_expected_norm_scaled,
    alpha=0.4,
    s=15,
    label="Log Expected (scaled)",
    color="brown",
)
axes[3, 1].set_xlabel("Sink Value")
axes[3, 1].set_ylabel("Normalized Log Matrix Norm")
axes[3, 1].set_title("Sink Value vs All Log Matrix Norms (Normalized)")
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].legend()

plt.tight_layout()
plt.savefig("sink_vs_matrix_norms.png", dpi=300, bbox_inches="tight")
plt.show()

# Calculate correlations
correlation_fro = np.corrcoef(sink_values, fro_norms)[0, 1]
correlation_spectral = np.corrcoef(sink_values, spectral_norms)[0, 1]
correlation_expected = np.corrcoef(sink_values, expected_norms)[0, 1]

# Calculate correlations with log norms
correlation_log_fro = np.corrcoef(sink_values, log_fro_norms)[0, 1]
correlation_log_spectral = np.corrcoef(sink_values, log_spectral_norms)[0, 1]
correlation_log_expected = np.corrcoef(sink_values, log_expected_norms)[0, 1]

print(f"\nCorrelations:")
print(f"Sink vs Frobenius norm: {correlation_fro:.4f}")
print(f"Sink vs Spectral norm: {correlation_spectral:.4f}")
print(f"Sink vs Expected norm: {correlation_expected:.4f}")
print(f"\nCorrelations with Log Norms:")
print(f"Sink vs Log Frobenius norm: {correlation_log_fro:.4f}")
print(f"Sink vs Log Spectral norm: {correlation_log_spectral:.4f}")
print(f"Sink vs Log Expected norm: {correlation_log_expected:.4f}")

# Save the raw data for further analysis if needed
data_dict = {
    "sink_values": sink_values.tolist(),
    "frobenius_norms": fro_norms.tolist(),
    "spectral_norms": spectral_norms.tolist(),
    "expected_norms": expected_norms.tolist(),
    "log_frobenius_norms": log_fro_norms.tolist(),
    "log_spectral_norms": log_spectral_norms.tolist(),
    "log_expected_norms": log_expected_norms.tolist(),
    "correlations": {
        "sink_vs_frobenius": correlation_fro,
        "sink_vs_spectral": correlation_spectral,
        "sink_vs_expected": correlation_expected,
        "sink_vs_log_frobenius": correlation_log_fro,
        "sink_vs_log_spectral": correlation_log_spectral,
        "sink_vs_log_expected": correlation_log_expected,
    },
}

with open("sink_vs_norms_data.json", "w") as f:
    json.dump(data_dict, f, indent=2)

print(f"\nAnalysis complete! Plot saved as 'sink_vs_matrix_norms.png'")
print(f"Raw data saved as 'sink_vs_norms_data.json'")
# %%
# Statistical significance test for correlation between sink values and expected norm
from scipy.stats import pearsonr
import scipy.stats as stats

# Perform Pearson correlation test with p-value
correlation_coeff, p_value = pearsonr(sink_values, expected_norms)

# One-tailed t-test for positive correlation
# H0: r <= 0 (negative or zero correlation)
# H1: r > 0 (positive correlation)
n = len(sink_values)
t_statistic = correlation_coeff * np.sqrt((n - 2) / (1 - correlation_coeff**2))
df = n - 2

# One-tailed p-value (testing for positive correlation)
p_value_one_tailed = 1 - stats.t.cdf(t_statistic, df)

print(f"\n--- Statistical Test for Sink vs Expected Norm Correlation ---")
print(f"Correlation coefficient: {correlation_coeff:.6f}")
print(f"Sample size: {n}")
print(f"Degrees of freedom: {df}")
print(f"T-statistic: {t_statistic:.6f}")
print(f"One-tailed p-value (H1: r > 0): {p_value_one_tailed:.6e}")
print(f"Two-tailed p-value: {p_value:.6e}")

# Interpret results
alpha = 0.05
if p_value_one_tailed < alpha:
    print(f"\nResult: SIGNIFICANT at α = {alpha}")
    print("We reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is significant evidence of a positive correlation between sink values and expected norm."
    )
else:
    print(f"\nResult: NOT SIGNIFICANT at α = {alpha}")
    print("We fail to reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is insufficient evidence of a positive correlation between sink values and expected norm."
    )

# Effect size interpretation
if abs(correlation_coeff) < 0.1:
    effect_size = "negligible"
elif abs(correlation_coeff) < 0.3:
    effect_size = "small"
elif abs(correlation_coeff) < 0.5:
    effect_size = "medium"
else:
    effect_size = "large"

print(f"Effect size: {effect_size} correlation")

# %%
# Statistical significance test for correlation between sink values and Frobenius norm
print(f"\n--- Statistical Test for Sink vs Frobenius Norm Correlation ---")

# Perform Pearson correlation test with p-value
correlation_coeff_fro, p_value_fro = pearsonr(sink_values, fro_norms)

# One-tailed t-test for positive correlation
t_statistic_fro = correlation_coeff_fro * np.sqrt(
    (n - 2) / (1 - correlation_coeff_fro**2)
)

# One-tailed p-value (testing for positive correlation)
p_value_one_tailed_fro = 1 - stats.t.cdf(t_statistic_fro, df)

print(f"Correlation coefficient: {correlation_coeff_fro:.6f}")
print(f"Sample size: {n}")
print(f"Degrees of freedom: {df}")
print(f"T-statistic: {t_statistic_fro:.6f}")
print(f"One-tailed p-value (H1: r > 0): {p_value_one_tailed_fro:.6e}")
print(f"Two-tailed p-value: {p_value_fro:.6e}")

# Interpret results
if p_value_one_tailed_fro < alpha:
    print(f"\nResult: SIGNIFICANT at α = {alpha}")
    print("We reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is significant evidence of a positive correlation between sink values and Frobenius norm."
    )
else:
    print(f"\nResult: NOT SIGNIFICANT at α = {alpha}")
    print("We fail to reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is insufficient evidence of a positive correlation between sink values and Frobenius norm."
    )

# Effect size interpretation
if abs(correlation_coeff_fro) < 0.1:
    effect_size_fro = "negligible"
elif abs(correlation_coeff_fro) < 0.3:
    effect_size_fro = "small"
elif abs(correlation_coeff_fro) < 0.5:
    effect_size_fro = "medium"
else:
    effect_size_fro = "large"

print(f"Effect size: {effect_size_fro} correlation")

# %%
# Statistical significance test for correlation between sink values and Spectral norm
print(f"\n--- Statistical Test for Sink vs Spectral Norm Correlation ---")

# Perform Pearson correlation test with p-value
correlation_coeff_spec, p_value_spec = pearsonr(sink_values, spectral_norms)

# One-tailed t-test for positive correlation
t_statistic_spec = correlation_coeff_spec * np.sqrt(
    (n - 2) / (1 - correlation_coeff_spec**2)
)

# One-tailed p-value (testing for positive correlation)
p_value_one_tailed_spec = 1 - stats.t.cdf(t_statistic_spec, df)

print(f"Correlation coefficient: {correlation_coeff_spec:.6f}")
print(f"Sample size: {n}")
print(f"Degrees of freedom: {df}")
print(f"T-statistic: {t_statistic_spec:.6f}")
print(f"One-tailed p-value (H1: r > 0): {p_value_one_tailed_spec:.6e}")
print(f"Two-tailed p-value: {p_value_spec:.6e}")

# Interpret results
if p_value_one_tailed_spec < alpha:
    print(f"\nResult: SIGNIFICANT at α = {alpha}")
    print("We reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is significant evidence of a positive correlation between sink values and spectral norm."
    )
else:
    print(f"\nResult: NOT SIGNIFICANT at α = {alpha}")
    print("We fail to reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is insufficient evidence of a positive correlation between sink values and spectral norm."
    )

# Effect size interpretation
if abs(correlation_coeff_spec) < 0.1:
    effect_size_spec = "negligible"
elif abs(correlation_coeff_spec) < 0.3:
    effect_size_spec = "small"
elif abs(correlation_coeff_spec) < 0.5:
    effect_size_spec = "medium"
else:
    effect_size_spec = "large"

print(f"Effect size: {effect_size_spec} correlation")

# %%
# Statistical significance test for correlation between sink values and Log Frobenius norm
print(f"\n--- Statistical Test for Sink vs Log Frobenius Norm Correlation ---")

# Perform Pearson correlation test with p-value
correlation_coeff_log_fro, p_value_log_fro = pearsonr(sink_values, log_fro_norms)

# One-tailed t-test for positive correlation
t_statistic_log_fro = correlation_coeff_log_fro * np.sqrt(
    (n - 2) / (1 - correlation_coeff_log_fro**2)
)

# One-tailed p-value (testing for positive correlation)
p_value_one_tailed_log_fro = 1 - stats.t.cdf(t_statistic_log_fro, df)

print(f"Correlation coefficient: {correlation_coeff_log_fro:.6f}")
print(f"Sample size: {n}")
print(f"Degrees of freedom: {df}")
print(f"T-statistic: {t_statistic_log_fro:.6f}")
print(f"One-tailed p-value (H1: r > 0): {p_value_one_tailed_log_fro:.6e}")
print(f"Two-tailed p-value: {p_value_log_fro:.6e}")

# Interpret results
if p_value_one_tailed_log_fro < alpha:
    print(f"\nResult: SIGNIFICANT at α = {alpha}")
    print("We reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is significant evidence of a positive correlation between sink values and log Frobenius norm."
    )
else:
    print(f"\nResult: NOT SIGNIFICANT at α = {alpha}")
    print("We fail to reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is insufficient evidence of a positive correlation between sink values and log Frobenius norm."
    )

# Effect size interpretation
if abs(correlation_coeff_log_fro) < 0.1:
    effect_size_log_fro = "negligible"
elif abs(correlation_coeff_log_fro) < 0.3:
    effect_size_log_fro = "small"
elif abs(correlation_coeff_log_fro) < 0.5:
    effect_size_log_fro = "medium"
else:
    effect_size_log_fro = "large"

print(f"Effect size: {effect_size_log_fro} correlation")

# %%
# Statistical significance test for correlation between sink values and Log Spectral norm
print(f"\n--- Statistical Test for Sink vs Log Spectral Norm Correlation ---")

# Perform Pearson correlation test with p-value
correlation_coeff_log_spec, p_value_log_spec = pearsonr(sink_values, log_spectral_norms)

# One-tailed t-test for positive correlation
t_statistic_log_spec = correlation_coeff_log_spec * np.sqrt(
    (n - 2) / (1 - correlation_coeff_log_spec**2)
)

# One-tailed p-value (testing for positive correlation)
p_value_one_tailed_log_spec = 1 - stats.t.cdf(t_statistic_log_spec, df)

print(f"Correlation coefficient: {correlation_coeff_log_spec:.6f}")
print(f"Sample size: {n}")
print(f"Degrees of freedom: {df}")
print(f"T-statistic: {t_statistic_log_spec:.6f}")
print(f"One-tailed p-value (H1: r > 0): {p_value_one_tailed_log_spec:.6e}")
print(f"Two-tailed p-value: {p_value_log_spec:.6e}")

# Interpret results
if p_value_one_tailed_log_spec < alpha:
    print(f"\nResult: SIGNIFICANT at α = {alpha}")
    print("We reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is significant evidence of a positive correlation between sink values and log spectral norm."
    )
else:
    print(f"\nResult: NOT SIGNIFICANT at α = {alpha}")
    print("We fail to reject the null hypothesis (H0: r ≤ 0)")
    print(
        "There is insufficient evidence of a positive correlation between sink values and log spectral norm."
    )

# Effect size interpretation
if abs(correlation_coeff_log_spec) < 0.1:
    effect_size_log_spec = "negligible"
elif abs(correlation_coeff_log_spec) < 0.3:
    effect_size_log_spec = "small"
elif abs(correlation_coeff_log_spec) < 0.5:
    effect_size_log_spec = "medium"
else:
    effect_size_log_spec = "large"

print(f"Effect size: {effect_size_log_spec} correlation")

# %%
