import torch
import torch.nn as nn
from ..functions.generator import fc_block


class RepresentationLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        super().__init__()

        # global attr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        user_rep_slice = self.mlp["user"](user_emb)
        item_rep_slice = self.mlp["item"](item_emb)
        return user_rep_slice, item_rep_slice

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        
        components = list(fc_block(**kwargs))
        mlp_user = nn.Sequential(*components)

        components = list(fc_block(**kwargs))
        mlp_item = nn.Sequential(*components)

        components = dict(
            user=mlp_user,
            item=mlp_item,
        )
        self.mlp = nn.ModuleDict(components)
