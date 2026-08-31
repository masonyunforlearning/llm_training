# Experiment plan

1. Profile every candidate corpus with >=10K samples.
2. Check quality, language purity, duplication, boilerplate, provenance and licensing.
3. Benchmark tokenizer efficiency on the actual Korean/English mixture.
4. Run 50M proxy ablations at 10/90, 20/80, 30/70 and 40/60.
5. Keep model, optimizer, schedule, seed, sequence length and total tokens fixed.
6. Report Korean and English validation losses separately.
7. Select the final mixture from measured results, not from an assumed ratio.
8. Only then start the 1.2B run.

Two GPUs do not automatically provide one combined VRAM pool; use an appropriate distributed/sharding strategy.
