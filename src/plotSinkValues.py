#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
#%%
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map='cpu', torch_dtype=torch.bfloat16)
#%%
# Extract sink values from all layers with strict validation
layer_numbers = []
sink_values = []

for i, layer in enumerate(model.model.layers):
    # Strict validation - fail loudly if format doesn't match
    if not hasattr(layer, 'self_attn'):
        raise AttributeError(f"Layer {i} does not have 'self_attn' attribute")
    
    if not hasattr(layer.self_attn, '_parameters'):
        raise AttributeError(f"Layer {i}.self_attn does not have '_parameters' attribute")
    
    if 'sinks' not in layer.self_attn._parameters:
        raise KeyError(f"Layer {i}.self_attn._parameters does not contain 'sinks' key")
    
    sinks = layer.self_attn._parameters['sinks']
    
    if sinks is None:
        raise ValueError(f"Layer {i}.self_attn._parameters['sinks'] is None")
    
    if not torch.is_tensor(sinks):
        raise TypeError(f"Layer {i}.self_attn._parameters['sinks'] is not a tensor, got {type(sinks)}")
    
    # Convert tensor to numpy and flatten
    sinks_np = sinks.float().detach().cpu().numpy().flatten()
    
    # Add layer number for each sink value
    layer_numbers.extend([i] * len(sinks_np))
    sink_values.extend(sinks_np)

# Prepare data for linear regression
X = np.array(layer_numbers).reshape(-1, 1)
y = np.array(sink_values)

# Fit linear regression
reg = LinearRegression()
reg.fit(X, y)

# Generate points for regression line
x_range = np.linspace(min(layer_numbers), max(layer_numbers), 100)
y_pred = reg.predict(x_range.reshape(-1, 1))

# Create the plot
plt.figure(figsize=(12, 8))
plt.scatter(layer_numbers, sink_values, alpha=0.6, s=10, label='Sink Values')

# Plot regression line
plt.plot(x_range, y_pred, 'red', linewidth=2, label=f'Linear Regression (slope: {reg.coef_[0]:.6f})')

plt.xlabel('Layer Number')
plt.ylabel('Sink Value')
plt.title('Sink Values Across All Layers with Linear Regression')
plt.grid(True, alpha=0.3)

# Add mean line
mean_val = np.mean(sink_values)
plt.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')

plt.legend()
plt.tight_layout()
plt.show()

# Print some statistics
print(f"Total layers processed: {len(set(layer_numbers))}")
print(f"Total sink values: {len(sink_values)}")
print(f"Mean sink value: {np.mean(sink_values):.4f}")
print(f"Std sink value: {np.std(sink_values):.4f}")
print(f"Min sink value: {np.min(sink_values):.4f}")
print(f"Max sink value: {np.max(sink_values):.4f}")
print(f"\nLinear Regression Results:")
print(f"Slope: {reg.coef_[0]:.6f}")
print(f"Intercept: {reg.intercept_:.6f}")
print(f"R² Score: {reg.score(X, y):.6f}")
# %%
