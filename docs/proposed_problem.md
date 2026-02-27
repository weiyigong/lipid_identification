# Lipid Identification from MS2 Fragmentation Spectra

## Background

Mass spectrometry-based lipidomics relies on tandem mass spectrometry (MS/MS or MS2) to identify lipid species. In an MS2 experiment, a precursor ion is selected and fragmented, producing a characteristic pattern of fragment ions. This fragmentation pattern serves as a "molecular fingerprint" that can be used to identify the original compound.

The **LipidBlast** database contains reference MS2 spectra for thousands of lipid species, making it an ideal resource for developing and benchmarking compound identification algorithms.

## Dataset Description

You are provided with a parquet file containing MS2 spectra from the LipidBlast 2022 database. Each record represents a single spectrum with the following fields:

| Field | Type | Description |
|---|---|---|
| `name` | string | Full lipid name (e.g., "PC 34:1", "TG 52:3") |
| `precursor_mz` | float | Mass-to-charge ratio of the precursor ion |
| `mz_list` | list[float] | Fragment ion m/z values |
| `intensity_list` | list[float] | Corresponding fragment ion intensities |
| `mode` | string | Ionization mode ("positive" or "negative") |
| `adduct_name` | string | Adduct type (e.g., "[M+H]+", "[M-H]-", "[M+Na]+") |

**Dataset characteristics:**

- Contains spectra across both positive and negative ionization modes
- Covers diverse lipid classes (PC, PE, TG, DG, SM, ceramides, etc.)
- Spectra vary in complexity (number of peaks ranges widely)
- Precursor m/z spans a broad range typical of lipidomics experiments
- Multiple adduct types are represented for many compounds

## Resources & Data Access

| Resource | Link |
|---|---|
| **Dataset (Parquet file)** | [MoNA-export-LipidBlast_2022.parquet](https://onedrive.live.com) *(OneDrive)* |
| **Data analysis script** | [dataset_analysis_basic.py](https://onedrive.live.com) *(OneDrive)* |

> *Note: The dataset is derived from LipidBlast synthetic spectra, chosen for their size and public availability. Some preprocessing has been applied to create the parquet format.*

## Your Task

Design and implement an algorithm that, given a **query MS2 spectrum**, identifies the most likely matching compound(s) from the reference library.

Your solution should:

1. **Accept a query spectrum** consisting of:
   - Precursor m/z
   - List of fragment m/z values
   - List of fragment intensities
   - Ionization mode

2. **Return ranked candidate matches** with confidence scores

3. **Handle real-world challenges** such as:
   - Mass measurement error/tolerance
   - Missing or noisy peaks
   - Intensity variations between instruments
   - Computational efficiency at scale

## Evaluation Criteria

Your submission will be evaluated on:

| Criterion | Weight | Description |
|---|---|---|
| **Accuracy** | 30% | Correct identification rate on held-out test spectra |
| **Elegance** | 25% | Clarity, simplicity, and cleverness of the approach |
| **Scalability** | 20% | Efficiency with large reference libraries (think 100k+ spectra) |
| **Robustness** | 15% | Performance under noise, missing peaks, and mass shifts |
| **Code Quality** | 10% | Readability, documentation, and software engineering practices |

## Suggested Approaches (Non-Exhaustive)

You are free to explore any method. Some starting points to consider:

- **Spectral similarity metrics:** Cosine similarity, dot product, entropy-based measures
- **Peak matching algorithms:** With configurable m/z tolerance windows
- **Machine learning:** Siamese networks, contrastive learning, spectral embeddings
- **Hybrid approaches:** Precursor filtering + spectral matching
- **Indexing strategies:** For efficient candidate retrieval at scale

## Deliverables

1. **GitHub repository:** Well-documented Python implementation of your identification algorithm
2. **Brief report (1-2 pages):** Explaining your approach, design decisions, and trade-offs
3. **Evaluation notebook/code:** Demonstrating your method's performance with metrics and visualizations
4. **Presentation (~30 min):** A brief walkthrough of your chosen approach, key design decisions, and results

## Bonus Challenges (Optional)

- Handle cases where the query spectrum may not exist in the reference library (open-set identification)
- Predict lipid class even when exact match isn't possible
- Quantify uncertainty in your predictions
- Optimize for sub-second query times on the full database

## Getting Started

The following snippet illustrates how we expect your `identify_compound()` function to be called. Your implementation should follow this interface:

```python
import pandas as pd

# Load the reference library
df = pd.read_parquet("MoNA-export-LipidBlast_2022.parquet")

# Example: Create a simulated query from the dataset
query_spectrum = df.sample(1).iloc[0]
query = {
    "precursor_mz": query_spectrum["precursor_mz"],
    "mz_list": query_spectrum["mz_list"],
    "intensity_list": query_spectrum["intensity_list"],
    "mode": query_spectrum["mode"]
}

# Your task: implement identify_compound()
# results = identify_compound(query, reference_library=df)
```

## Time Expectation

You will have **one week** from receiving this challenge to submit your deliverables. The exercise itself is designed to take approximately **3-4 hours**. We value thoughtful solutions over exhaustive ones—focus on demonstrating your problem-solving approach and scientific reasoning.

## Resources

- [LipidBlast documentation](https://fiehnlab.ucdavis.edu/projects/lipidblast)
- Relevant literature on spectral matching:
  - **LSM‑MS2: A Foundation Model Bridging Spectral Identification and Biological Interpretation** (arXiv, 2025)[1]
