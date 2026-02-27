"""ChemBERTa molecular encoder for cross-modal alignment."""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class ChemBERTaEncoder:
    """Frozen ChemBERTa: SMILES -> CLS token -> L2-normalized 768-dim embedding."""

    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.d_mol = self.model.config.hidden_size

    def encode_smiles(self, smiles):
        """Single SMILES -> (d_mol,) numpy array."""
        tokens = self.tokenizer(smiles, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens).last_hidden_state[:, 0]
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out.cpu().numpy().squeeze(0)

    def encode_batch(self, smiles_list, batch_size=128):
        """Batch SMILES -> (N, d_mol) numpy array."""
        all_embs = []
        for start in tqdm(range(0, len(smiles_list), batch_size),
                          desc="ChemBERTa encoding", leave=False):
            batch = smiles_list[start:start + batch_size]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out = self.model(**tokens).last_hidden_state[:, 0]
                out = torch.nn.functional.normalize(out, p=2, dim=-1)
            all_embs.append(out.cpu().numpy())
        return np.concatenate(all_embs, axis=0)
