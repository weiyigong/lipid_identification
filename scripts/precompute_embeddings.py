"""Precompute frozen teacher embeddings (DreaMS + ChemBERTa) for all reference spectra."""

from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def precompute_dreams(df, cache_path, batch_size=256):
    """Precompute DreaMS embeddings for all spectra."""
    if cache_path.exists():
        embs = np.load(cache_path)
        print(f"DreaMS embeddings already cached: {embs.shape}")
        return embs

    from src.models.dreams_pipeline import _embed_spectra, _normalize

    print(f"Computing DreaMS embeddings for {len(df):,} spectra...")
    from dreams.api import PreTrainedModel
    model = PreTrainedModel.from_name("DreaMS_embedding")

    embs = _embed_spectra(
        model,
        df["mz_list"].tolist(),
        df["intensity_list"].tolist(),
        df["precursor_mz"].tolist(),
        batch_size=batch_size,
        progress_bar=True,
    )
    embs = _normalize(embs)
    np.save(cache_path, embs)
    print(f"Saved DreaMS embeddings: {embs.shape} -> {cache_path}")
    return embs


def precompute_chemberta(df, cache_path, batch_size=128):
    """Precompute ChemBERTa embeddings for all SMILES."""
    if cache_path.exists():
        embs = np.load(cache_path)
        print(f"ChemBERTa embeddings already cached: {embs.shape}")
        return embs

    from src.models.mol_encoder import ChemBERTaEncoder
    encoder = ChemBERTaEncoder()

    smiles_col = "isomeric_smiles" if "isomeric_smiles" in df.columns else "smiles"
    smiles_list = df[smiles_col].tolist()
    print(f"Computing ChemBERTa embeddings for {len(smiles_list):,} SMILES...")

    embs = encoder.encode_batch(smiles_list, batch_size=batch_size)
    np.save(cache_path, embs)
    print(f"Saved ChemBERTa embeddings: {embs.shape} -> {cache_path}")
    return embs


def main():
    import sys
    from src.data.loader import load_library

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_library()

    chemberta_path = CACHE_DIR / "chemberta_ref_embeddings.npy"
    dreams_path = CACHE_DIR / "dreams_ref_embeddings.npy"

    # ChemBERTa is always needed; DreaMS only with --dreams flag
    precompute_chemberta(df, chemberta_path)

    if "--dreams" in sys.argv:
        precompute_dreams(df, dreams_path)
    elif not dreams_path.exists():
        print("Skipping DreaMS (slow). Pass --dreams to compute, or train without distillation.")

    print("Done.")


if __name__ == "__main__":
    main()
