import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_factors: int,
        hidden: list,
        dropout: float,
    ):
        super(Module, self).__init__()
        # attr dictionary for load

        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_embed: torch.Tensor, 
        item_embed: torch.Tensor,
    ):
        kwargs = dict(
            tensors=(user_embed, item_embed), 
            dim=-1,
        )
        concat = torch.cat(**kwargs)
        pred_vector = self.mlp_layers(concat)
        return pred_vector 

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        components = list(self._yield_layers(self.hidden))
        self.mlp_layers = nn.Sequential(*components)

    def _yield_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 2)
        ERROR_MESSAGE = f"First matching function layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE