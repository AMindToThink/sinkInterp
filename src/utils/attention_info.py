import torch


def get_head_projections(model, layer_idx, head_idx):
    """
    Get all projection parameters (q, k, v) for a specific attention head.

    Returns the parameters as tensors that can be used for analysis or modification.
    """
    attention = model.model.layers[layer_idx].self_attn
    config = model.config

    # Get configuration
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = attention.head_dim
    num_key_value_groups = num_attention_heads // num_key_value_heads

    # Validate head index
    if head_idx >= num_attention_heads:
        raise IndexError(
            f"Head {head_idx} doesn't exist. Layer has {num_attention_heads} heads."
        )

    # Calculate query head slice
    q_start = head_idx * head_dim
    q_end = (head_idx + 1) * head_dim

    # Calculate key/value head slice (grouped query attention)
    kv_head_idx = head_idx // num_key_value_groups
    kv_start = kv_head_idx * head_dim
    kv_end = (kv_head_idx + 1) * head_dim

    # Extract parameters
    q_weight = attention.q_proj.weight[q_start:q_end, :]
    k_weight = attention.k_proj.weight[kv_start:kv_end, :]
    v_weight = attention.v_proj.weight[kv_start:kv_end, :]

    result = {
        "q_proj_weight": q_weight,
        "k_proj_weight": k_weight,
        "v_proj_weight": v_weight,
        "head_idx": head_idx,
        "kv_head_idx": kv_head_idx,
        "q_slice": (q_start, q_end),
        "kv_slice": (kv_start, kv_end),
        "sink": attention.sinks[head_idx],
    }

    # Add bias if it exists
    if attention.q_proj.bias is not None:
        result["q_proj_bias"] = attention.q_proj.bias[q_start:q_end]
        result["k_proj_bias"] = attention.k_proj.bias[kv_start:kv_end]
        result["v_proj_bias"] = attention.v_proj.bias[kv_start:kv_end]

    return result


def findQuadraticForm(result):
    """
    Compute the quadratic form Q^T K for attention computation.

    If bias terms exist, this computes the full affine transformation
    by treating the input as homogeneous coordinates.
    """
    q_weight = result["q_proj_weight"]  # (head_dim, hidden_size)
    k_weight = result["k_proj_weight"]  # (head_dim, hidden_size)

    # Check if bias terms exist
    if "q_proj_bias" not in result or "k_proj_bias" not in result:
        raise ValueError(
            "Bias terms not found in result. Cannot compute projective form."
        )

    q_bias = result["q_proj_bias"]  # (head_dim,)
    k_bias = result["k_proj_bias"]  # (head_dim,)

    # Validate dimensions
    assert (
        q_weight.shape[0] == q_bias.shape[0]
    ), "Q weight and bias head dimensions must match"
    assert (
        k_weight.shape[0] == k_bias.shape[0]
    ), "K weight and bias head dimensions must match"
    assert (
        q_weight.shape[1] == k_weight.shape[1]
    ), "Q and K hidden dimensions must match"

    # Create projective transformation matrices
    # Add bias as an additional column (for homogeneous coordinates)
    projectiveQ = torch.cat(
        [q_weight, q_bias.unsqueeze(1)], dim=1
    )  # (head_dim, hidden_size + 1)
    projectiveK = torch.cat(
        [k_weight, k_bias.unsqueeze(1)], dim=1
    )  # (head_dim, hidden_size + 1)

    # Compute quadratic form
    return projectiveQ @ projectiveK.T  # (head_dim, head_dim)


def expected_quadratic_form_norm(M):
    """Assumptions built into this norm:
    - No covariance in x
    - x is mean 0 (except because there's a bias term in W_q and W_k, the last index is 1)
    """
    assert M.shape.__len__() == 2
    bias_value = M[-1, -1]
    trace_value = torch.trace(
        M.float()
    )  # Trace is not implemented for torch.bfloat16, so cast to float
    return trace_value + bias_value


# def getHead(model, layer, idx, num_attention_heads, num_key_value_heads):
#     attention = model.model.layers[layer].self_attn
#     assert num_attention_heads % num_key_value_heads == 0
#     num_key_value_groups = num_attention_heads // num_key_value_heads
#     keyIndex = idx // num_key_value_groups
#     q_tensor = attention.q_proj.weight
#     k_tensor = attention.k_proj.weight
#     v_tensor = attention.v_proj.weight
#     return Head(query=, key=attention.k_proj, value=attention.v_proj)
