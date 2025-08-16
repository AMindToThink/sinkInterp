# Placeholder context manager for head ablation
from contextlib import contextmanager
import torch


def get_sink(model, layer, head):
    """Get sink value for a specific layer and head."""
    if layer >= len(model.model.layers):
        raise IndexError(
            f"Layer {layer} does not exist. Model has {len(model.model.layers)} layers."
        )

    layer_obj = model.model.layers[layer]
    if not hasattr(layer_obj, "self_attn"):
        raise AttributeError(f"Layer {layer} does not have 'self_attn' attribute")

    if not hasattr(layer_obj.self_attn, "_parameters"):
        raise AttributeError(
            f"Layer {layer}.self_attn does not have '_parameters' attribute"
        )

    if "sinks" not in layer_obj.self_attn._parameters:
        raise KeyError(
            f"Layer {layer}.self_attn._parameters does not contain 'sinks' key"
        )

    sinks = layer_obj.self_attn._parameters["sinks"]
    if head >= len(sinks):
        raise IndexError(
            f"Head {head} does not exist. Layer {layer} has {len(sinks)} heads."
        )

    return sinks[head]


@contextmanager
def ablate_head(model, layer, head):
    """
    Deactivates a head with a sink and then reactivates it.
    Sets the sink to infinity to capture all attention from tokens.
    """
    print(f"  Ablating layer {layer}, head {head}")

    # Validate inputs and get sink parameters
    sink_params = model.model.layers[layer].self_attn._parameters["sinks"]
    if head >= len(sink_params):
        raise IndexError(
            f"Head {head} does not exist. Layer {layer} has {len(sink_params)} heads."
        )

    original_value = sink_params[head].clone()

    try:
        with torch.no_grad():
            # Create infinity tensor with same shape/device/dtype as original
            inf_value = torch.full_like(
                original_value, 100000000
            )  # Can't be actual inf because that results in nan values.
            sink_params[head] = inf_value
        yield

    finally:
        # Always restore original value, even if an exception occurred
        with torch.no_grad():
            sink_params[head] = original_value
        print(f"  Restoring layer {layer}, head {head}")
