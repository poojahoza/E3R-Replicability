from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel
import warnings

class ProjectionScoreModel(nn.Module):
    def __init__(self, input_emb_dim: int, output_emb_dim: int):
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.output_emb_dim = output_emb_dim
        self.projection = nn.Bilinear(in1_features=input_emb_dim, in2_features=input_emb_dim, out_features=output_emb_dim)
        self.classifier = nn.Linear(output_emb_dim, 1)

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor):
        return self.classifier(self.projection(query_emb, entity_emb))


