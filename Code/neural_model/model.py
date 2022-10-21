from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel
import warnings

class ProjectionScoreModel(nn.Module):
    def __init__(self, input_emb_dim: int):
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.classifier = nn.Linear(input_emb_dim, 1)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor):
        mult_emb = torch.mm(query_emb, entity_emb)
        return self.classifier(mult_emb)


