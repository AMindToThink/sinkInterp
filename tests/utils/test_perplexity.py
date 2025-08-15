import pytest
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import sys
import os

# Add the src directory to the path so we can import our modified perplexity module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from utils.perplexity import Perplexity


class TestPerplexityEquivalence:
    """Test that our modified perplexity implementation produces identical results to the original."""

    @pytest.mark.model
    @pytest.fixture(scope="class")
    def models_and_tokenizers(self):
        """Load models and tokenizers for testing."""
        models = {}
        tokenizers = {}

        # Test with multiple models to ensure compatibility
        model_ids = ["google/gemma-2-2b"]

        for model_id in model_ids:
            models[model_id] = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
            tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)

        return models, tokenizers

    @pytest.fixture(scope="class")
    def original_perplexity(self):
        """Load the original perplexity metric from evaluate."""
        return evaluate.load("perplexity", module_type="metric")

    @pytest.fixture(scope="class")
    def modified_perplexity(self):
        """Create instance of our modified perplexity metric."""
        return Perplexity()

    @pytest.fixture
    def test_texts(self):
        """Sample texts for testing - same as in perplexityMeasure.py plus additional cases."""
        return [
            # Original texts from perplexityMeasure.py
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
            "Happy Birthday to you!",
            "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.",
            # Additional test cases
            "The quick brown fox jumps over the lazy dog.",
            "A",  # Single character
            "Hello world! This is a test.",
            "Python is a programming language.",
        ]

    @pytest.mark.gpu
    def test_identical_results_gemma(
        self,
        models_and_tokenizers,
        original_perplexity,
        modified_perplexity,
        test_texts,
    ):
        """Test that both implementations produce identical results with Gemma-2-2B."""
        models, tokenizers = models_and_tokenizers
        model_id = "google/gemma-2-2b"
        model = models[model_id]
        tokenizer = tokenizers[model_id]

        # Test with default parameters
        original_results = original_perplexity.compute(
            model_id=model_id, add_start_token=True, predictions=test_texts
        )

        modified_results = modified_perplexity.compute(
            model=model,
            tokenizer=tokenizer,
            add_start_token=True,
            predictions=test_texts,
        )
        print("Original results: ", original_results)
        print("Modified results: ", modified_results)
        # Check that results are identical
        assert np.allclose(
            original_results["perplexities"],
            modified_results["perplexities"],
            rtol=1e-10,
            atol=1e-10,
        )
        assert np.allclose(
            original_results["mean_perplexity"],
            modified_results["mean_perplexity"],
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.slow
    def test_different_batch_sizes(
        self,
        models_and_tokenizers,
        original_perplexity,
        modified_perplexity,
        test_texts,
    ):
        """Test that results are identical with different batch sizes."""
        models, tokenizers = models_and_tokenizers
        model_id = "google/gemma-2-2b"
        model = models[model_id]
        tokenizer = tokenizers[model_id]

        for batch_size in [1, 2, 4]:
            original_results = original_perplexity.compute(
                model_id=model_id,
                add_start_token=True,
                predictions=test_texts,
                batch_size=batch_size,
            )

            modified_results = modified_perplexity.compute(
                model=model,
                tokenizer=tokenizer,
                add_start_token=True,
                predictions=test_texts,
                batch_size=batch_size,
            )

            # Check that results are identical
            assert np.allclose(
                original_results["perplexities"],
                modified_results["perplexities"],
                rtol=1e-10,
                atol=1e-10,
            )
            assert np.allclose(
                original_results["mean_perplexity"],
                modified_results["mean_perplexity"],
                rtol=1e-10,
                atol=1e-10,
            )

    def test_without_start_token(
        self,
        models_and_tokenizers,
        original_perplexity,
        modified_perplexity,
        test_texts,
    ):
        """Test that results are identical when add_start_token=False."""
        models, tokenizers = models_and_tokenizers
        model_id = "google/gemma-2-2b"
        model = models[model_id]
        tokenizer = tokenizers[model_id]

        # Filter out single character texts as they require at least 2 tokens when add_start_token=False
        longer_texts = [text for text in test_texts if len(text.split()) > 1]

        original_results = original_perplexity.compute(
            model_id=model_id, add_start_token=False, predictions=longer_texts
        )

        modified_results = modified_perplexity.compute(
            model=model,
            tokenizer=tokenizer,
            add_start_token=False,
            predictions=longer_texts,
        )

        # Check that results are identical
        assert np.allclose(
            original_results["perplexities"],
            modified_results["perplexities"],
            rtol=1e-10,
            atol=1e-10,
        )
        assert np.allclose(
            original_results["mean_perplexity"],
            modified_results["mean_perplexity"],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_with_max_length(
        self, models_and_tokenizers, original_perplexity, modified_perplexity
    ):
        """Test that results are identical when max_length is specified."""
        models, tokenizers = models_and_tokenizers
        model_id = "google/gemma-2-2b"
        model = models[model_id]
        tokenizer = tokenizers[model_id]

        # Use a long text that will be truncated
        long_texts = [
            "This is a very long text that should be truncated when max_length is set. "
            * 20
        ]

        max_length = 50

        original_results = original_perplexity.compute(
            model_id=model_id,
            add_start_token=True,
            predictions=long_texts,
            max_length=max_length,
        )

        modified_results = modified_perplexity.compute(
            model=model,
            tokenizer=tokenizer,
            add_start_token=True,
            predictions=long_texts,
            max_length=max_length,
        )

        # Check that results are identical
        assert np.allclose(
            original_results["perplexities"],
            modified_results["perplexities"],
            rtol=1e-10,
            atol=1e-10,
        )
        assert np.allclose(
            original_results["mean_perplexity"],
            modified_results["mean_perplexity"],
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])
