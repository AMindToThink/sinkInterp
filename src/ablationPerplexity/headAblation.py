# %%
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import perplexity
import torch
torch.cuda.empty_cache()

# %%
input_texts = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[
    "text"
]
input_texts = [s for s in input_texts if s != ""][:50]

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
# %%
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", torch_dtype=torch.bfloat16).to("cuda")
# %%
results = perplexity.Perplexity().compute(
    model=model,
    tokenizer=tokenizer,
    predictions=input_texts,
)
# %%
print(results)
