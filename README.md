# sinkInterp

In progress. Only made public early because it is easier than to deal with GitHub permissions, and I wanted to clone it to a vast.ai machine for more experiments.

An interpretability research project studying the learned per-head attention "sink" values in
`openai/gpt-oss-20b`. The scripts in `src/` extract each head's sink parameter and correlate it
against that head's query/key weight-matrix norms (Frobenius, spectral, and an "expected
quadratic form" norm), split analyses by even vs. odd layer, and run a per-head ablation sweep
(replacing a head's sink with a very large value) to measure the resulting perplexity change on
`wikitext-2-raw-v1`. Results (JSON data and plots) are checked into `src/` and `plots/`.

See **[REPRODUCE.md](REPRODUCE.md)** for exact environment setup, data sources, entry-point
commands, external dependencies, and hardware requirements.
