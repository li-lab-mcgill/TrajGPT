import time
import pandas as pd
import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbeddingLearnable(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbeddingLearnable, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TokenEmbeddingFixed(nn.Module):
    """Fixed vectors with an explicit, checkpointable raw-PheCode vocabulary.

    Raw code 0 denotes PAD. SOS (-1) is handled by TrajGPT, not by this table.
    Unknown codes are errors rather than silently being treated as padding.
    """

    def __init__(self, embedding_path=None, embedding_data=None):
        super(TokenEmbeddingFixed, self).__init__()
        if embedding_data is None:
            embedding_df = pd.read_csv(embedding_path, index_col=0)
            codes = embedding_df.index.tolist()
            embeddings = torch.tensor(embedding_df.values, dtype=torch.float32)
        else:
            codes = list(embedding_data['phecodes'])
            embeddings = torch.as_tensor(embedding_data['embeddings'], dtype=torch.float32).detach().clone()
        if any(float(code) != int(code) or int(code) <= 0 for code in codes):
            raise ValueError('The vocabulary must contain positive integer PheCodes; 0 is reserved for PAD.')
        self.phecodes = [int(code) for code in codes]
        if not self.phecodes or len(set(self.phecodes)) != len(self.phecodes):
            raise ValueError('PheCode vocabulary must be nonempty and contain no duplicates.')
        if embeddings.ndim != 2 or embeddings.size(0) != len(self.phecodes):
            raise ValueError('Embedding rows must match the PheCode vocabulary.')
        if not torch.isfinite(embeddings).all():
            raise ValueError('Embedding values must be finite.')
        self.phecode_to_index = {code: idx for idx, code in enumerate(self.phecodes)}
        self.padding_idx = len(self.phecodes)
        # A small dense lookup avoids Python loops and CPU copies per forward.
        lookup = torch.full((max(self.phecodes) + 1,), -1, dtype=torch.long)
        lookup[0] = self.padding_idx
        lookup[torch.tensor(self.phecodes)] = torch.arange(self.padding_idx)
        self.register_buffer('_code_lookup', lookup, persistent=False)
        embeddings = torch.cat([embeddings, embeddings.new_zeros(1, embeddings.size(1))], dim=0)
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=self.padding_idx)

    def get_num_tokens(self):
        return len(self.phecodes)

    def map_phecodes_to_indices(self, phecodes):
        phecodes = torch.as_tensor(phecodes, device=self._code_lookup.device)
        if phecodes.is_floating_point() and not torch.equal(phecodes, phecodes.trunc()):
            raise ValueError('PheCodes must be integer-valued.')
        codes = phecodes.long()
        in_range = (codes >= 0) & (codes < self._code_lookup.numel())
        safe_codes = codes.clamp(0, self._code_lookup.numel() - 1)
        indices = self._code_lookup[safe_codes]
        if not torch.all(in_range & (indices >= 0)):
            raise ValueError('Unknown PheCode: inputs and targets must use the saved vocabulary (PAD=0).')
        return indices

    def forward(self, x):
        return self.emb(self.map_phecodes_to_indices(x))
