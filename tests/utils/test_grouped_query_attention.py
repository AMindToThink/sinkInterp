import pytest
import torch
from transformers import AutoModelForCausalLM
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestGroupedQueryAttention:
    """Test the mapping between query projections and key/value projections in grouped query attention."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load the GPT-OSS model for testing."""
        return AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b", torch_dtype=torch.bfloat16
        )

    def test_projection_dimensions(self, model):
        """Test that projection dimensions match expected grouped query attention configuration."""
        layer = model.model.layers[0].self_attn
        config = model.config

        # Verify basic configuration
        assert hasattr(
            config, "num_attention_heads"
        ), "Config should have num_attention_heads"
        assert hasattr(
            config, "num_key_value_heads"
        ), "Config should have num_key_value_heads"

        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = layer.head_dim
        hidden_size = config.hidden_size

        # From the codebase analysis, we know GPT-OSS-20B has 64 attention heads
        assert (
            num_attention_heads == 64
        ), f"Expected 64 attention heads, got {num_attention_heads}"

        # Verify projection dimensions
        assert layer.q_proj.out_features == num_attention_heads * head_dim
        assert layer.k_proj.out_features == num_key_value_heads * head_dim
        assert layer.v_proj.out_features == num_key_value_heads * head_dim

        # Verify grouping relationship
        assert (
            num_attention_heads % num_key_value_heads == 0
        ), "num_attention_heads must be divisible by num_key_value_heads"
        assert layer.num_key_value_groups == num_attention_heads // num_key_value_heads

    def test_query_key_mapping(self, model):
        """
        Test that query projections are correctly mapped to key projections.

        In grouped query attention:
        - Query heads are divided into groups
        - Each group shares the same key and value heads
        - The repeat_kv function replicates k/v heads to match the number of query heads
        """
        layer = model.model.layers[0].self_attn
        config = model.config

        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        num_key_value_groups = layer.num_key_value_groups
        head_dim = layer.head_dim

        # Create sample input
        batch_size = 2
        seq_len = 10
        hidden_size = config.hidden_size
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, dtype=torch.bfloat16
        )

        # Forward through projections
        with torch.no_grad():
            query_states = layer.q_proj(hidden_states)
            key_states = layer.k_proj(hidden_states)
            value_states = layer.v_proj(hidden_states)

        # Reshape to separate heads
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        query_states = query_states.view(hidden_shape).transpose(
            1, 2
        )  # [batch, num_attention_heads, seq_len, head_dim]
        key_states = key_states.view(
            (*input_shape, num_key_value_heads, head_dim)
        ).transpose(
            1, 2
        )  # [batch, num_key_value_heads, seq_len, head_dim]
        value_states = value_states.view(
            (*input_shape, num_key_value_heads, head_dim)
        ).transpose(
            1, 2
        )  # [batch, num_key_value_heads, seq_len, head_dim]

        # Apply repeat_kv to match query heads
        from transformers.models.gpt_oss.modeling_gpt_oss import repeat_kv

        expanded_key_states = repeat_kv(key_states, num_key_value_groups)
        expanded_value_states = repeat_kv(value_states, num_key_value_groups)

        # Verify shapes
        assert query_states.shape == (
            batch_size,
            num_attention_heads,
            seq_len,
            head_dim,
        )
        assert expanded_key_states.shape == (
            batch_size,
            num_attention_heads,
            seq_len,
            head_dim,
        )
        assert expanded_value_states.shape == (
            batch_size,
            num_attention_heads,
            seq_len,
            head_dim,
        )

        # Test the specific mapping: which query heads correspond to which key heads
        for query_head_idx in range(num_attention_heads):
            # Calculate which key/value head this query head should use
            key_value_head_idx = query_head_idx // num_key_value_groups

            # Extract the corresponding slices
            query_head = query_states[:, query_head_idx, :, :]
            expected_key_head = key_states[:, key_value_head_idx, :, :]
            actual_key_head = expanded_key_states[:, query_head_idx, :, :]

            # Verify they match
            assert torch.allclose(
                expected_key_head, actual_key_head, rtol=1e-6
            ), f"Query head {query_head_idx} should map to key/value head {key_value_head_idx}"

    def test_attention_head_triples(self, model):
        """
        Test the creation of (q_proj, k_proj, v_proj) triples for each attention head.
        This verifies that we can correctly identify which slices of projections correspond to each head.
        """
        layer = model.model.layers[0].self_attn
        config = model.config

        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        num_key_value_groups = layer.num_key_value_groups
        head_dim = layer.head_dim

        # Create triples for each attention head
        triples = []

        for query_head_idx in range(num_attention_heads):
            # Calculate which key/value head this query head uses
            key_value_head_idx = query_head_idx // num_key_value_groups

            # Define the slices for projections
            q_start = query_head_idx * head_dim
            q_end = (query_head_idx + 1) * head_dim

            kv_start = key_value_head_idx * head_dim
            kv_end = (key_value_head_idx + 1) * head_dim

            triple = {
                "head_idx": query_head_idx,
                "q_proj_slice": (q_start, q_end),
                "k_proj_slice": (kv_start, kv_end),
                "v_proj_slice": (kv_start, kv_end),
                "key_value_head_idx": key_value_head_idx,
            }
            triples.append(triple)

        # Verify we have the right number of triples
        assert len(triples) == num_attention_heads

        # Verify that multiple query heads share the same key/value slices
        key_value_usage = {}
        for triple in triples:
            kv_head_idx = triple["key_value_head_idx"]
            if kv_head_idx not in key_value_usage:
                key_value_usage[kv_head_idx] = []
            key_value_usage[kv_head_idx].append(triple["head_idx"])

        # Each key/value head should be used by exactly num_key_value_groups query heads
        assert len(key_value_usage) == num_key_value_heads
        for kv_head_idx, query_heads in key_value_usage.items():
            assert (
                len(query_heads) == num_key_value_groups
            ), f"Key/value head {kv_head_idx} should be used by {num_key_value_groups} query heads, but is used by {len(query_heads)}"

        # Verify sequential grouping (query heads 0-3 use kv head 0, 4-7 use kv head 1, etc.)
        for kv_head_idx, query_heads in key_value_usage.items():
            expected_query_heads = list(
                range(
                    kv_head_idx * num_key_value_groups,
                    (kv_head_idx + 1) * num_key_value_groups,
                )
            )
            assert (
                sorted(query_heads) == expected_query_heads
            ), f"Key/value head {kv_head_idx} should be used by query heads {expected_query_heads}, but is used by {sorted(query_heads)}"

    def test_sink_parameter_mapping(self, model):
        """
        Test that sink parameters correctly map to attention heads.
        Each attention head should have exactly one sink parameter.
        """
        layer = model.model.layers[0].self_attn
        config = model.config

        num_attention_heads = config.num_attention_heads

        # Verify sink parameters exist and have correct shape
        assert hasattr(layer, "sinks"), "Layer should have sinks parameter"
        assert layer.sinks.shape == (
            num_attention_heads,
        ), f"Sinks should have shape ({num_attention_heads},), got {layer.sinks.shape}"

        # Each attention head gets its own sink value
        for head_idx in range(num_attention_heads):
            sink_value = layer.sinks[head_idx]
            assert isinstance(
                sink_value.item(), float
            ), f"Sink {head_idx} should be a scalar float"


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])
