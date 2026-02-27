# Lipid Identification from MS2 Spectra

Identifies lipid compounds from tandem mass spectra by matching against a 554K-spectrum LipidBlast reference library. Built as an ablation study: four methods of increasing complexity, benchmarked across 5 noise levels and 3 isomer-difficulty tiers.

Clean matching is trivial (cosine gets 100%). The real problem is degradation under noise, isomeric ambiguity, and doing it fast.

## Results

Top-1 accuracy on the 0% isomer eval set (500 queries, no spectrally identical isomers):

| Method | Clean | Mild | Moderate | Severe | Extreme | Query time |
|---|---|---|---|---|---|---|
| Cosine (classical) | 1.00 | 0.99 | 0.96 | 0.79 | 0.43 | 30 ms |
| DreaMS (pretrained) | 1.00 | 0.88 | 0.69 | 0.47 | 0.24 | 628 ms |
| Graph encoder (custom) | 1.00 | 0.70 | 0.73 | 0.57 | 0.28 | 0.7 ms |
| **Reranker (cross-attn)** | **1.00** | **0.99** | **0.97** | **0.83** | **0.42** | **39 ms** |

The reranker wins. It matches or beats cosine at every noise level and runs at comparable speed.

Bi-encoder approaches (DreaMS, graph encoder) underperform cosine on LipidBlast. The reason: with only ~5 peaks per spectrum and categorical intensities, a single embedding vector can't preserve the fine-grained peak positions that late-interaction methods keep. Cross-attention sidesteps this entirely.

## How it works

**Ablation ladder** — each method builds on what the previous one got wrong:

1. **Cosine similarity** — precursor filter (+-0.5 Da) narrows 554K to ~hundreds, then modified cosine with 0.02 Da fragment tolerance. The accuracy floor.

2. **DreaMS spectrum-to-spectrum** — frozen pretrained transformer encoder (Pluskal Lab), FAISS retrieval in 1024-d embedding space. Tests whether general-purpose spectral representations transfer to lipid retrieval. They don't, particularly.

3. **Spectral graph encoder** (~1 hr training) — peaks as graph nodes, mass differences as edges, 6-layer dense attention transformer (5.1M params). Multi-teacher training: ChemBERTa molecular alignment + DreaMS distillation + contrastive learning. Fastest method at 0.7 ms/query. Good class-level accuracy, but exact-match falls off under noise.

4. **Cross-attention reranker** (~30 min training) — takes cosine's top-50 candidates, scores each (query, candidate) pair with multi-head cross-attention augmented by DreaMS embeddings. Direct peak-to-peak comparison with learned importance weights.

## Key data insight

LipidBlast's 554K spectra collapse to ~423 unique intensity templates. Only m/z positions differ. ~65% of spectra have spectrally identical isomers (same formula + adduct, identical fragments). No spectral method can distinguish these — you'd need retention time or ion mobility. All methods converge to ~40% top-1 on 100% isomer queries, even clean.

## Quick start

```bash
# Install with uv
uv sync

# DreaMS (editable, for pretrained encoder)
uv pip install -e ./DreaMS
```

```python
from src.data.loader import load_library
from src.identify import identify_compound

ref = load_library()
query = ref.iloc[0]  # or any dict with mz_list, intensity_list, precursor_mz, mode

results = identify_compound(query, ref, top_k=10)
# Returns: [{"name": ..., "score": ..., "precursor_mz": ...}, ...]
```

`identify_compound` uses the cross-attention reranker if its checkpoint is present, otherwise falls back to cosine.

## Evaluation

Three eval sets at different isomer difficulty (0%, 50%, 100% isomers), each augmented at 5 noise levels (clean, mild, moderate, severe, extreme). Noise includes Gaussian m/z jitter, intensity scaling, stochastic peak dropout, and random noise peaks. Compound-level splits prevent leakage.

Metrics: top-K accuracy, MRR, class-level accuracy, query latency.

## Project structure

```
src/
  identify.py             # Public API: identify_compound()
  benchmark.py            # run_benchmark() across methods and eval sets
  constants.py            # Diagnostic ions, neutral losses (sourced)
  data/                   # Loader, splits, augmentation, eval set builders
  models/
    classical.py          # Cosine, modified cosine, entropy
    dreams_encoder.py     # Copied DreaMS architecture + weight loading
    dreams_pipeline.py    # FAISS retrieval in DreaMS embedding space
    spectral_graph_encoder.py  # Custom graph attention encoder
    reranker.py           # Cross-attention reranker
  train/                  # Training scripts for encoder and reranker
  utils/                  # Spectrum processing, indexing, metrics
notebooks/                # EDA, baseline benchmarks, training logs, final comparison
docs/report.md            # Full technical report
```

## Requirements

Python >=3.11, <3.13. Core dependencies: PyTorch, NumPy, pandas, FAISS, scikit-learn, RDKit, matchms. Runs on CPU or single GPU. All custom training completes in <2 hours on an RTX 3080.
