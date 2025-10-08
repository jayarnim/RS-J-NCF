import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        super(Module, self).__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_embed_slice = self.user_hist_embed_generator(user_idx, item_idx)
        item_embed_slice = self.item_hist_embed_generator(user_idx, item_idx)
        return user_embed_slice, item_embed_slice

    def user_hist_embed_generator(self, user_idx, item_idx):
        # get user vector from interactions
        user_interaction_slice = self.interactions[user_idx, :-1].clone()

        # masking target items
        user_idx_batch = torch.arange(user_idx.size(0))
        user_interaction_slice[user_idx_batch, item_idx] = 0

        # projection
        user_proj_slice = self.proj_u(user_interaction_slice.float())

        # representation learning
        user_rep_slice = self.rep_u(user_proj_slice)

        return user_rep_slice

    def item_hist_embed_generator(self, user_idx, item_idx):
        # get item vector from interactions
        item_interaction_slice = self.interactions.T[item_idx, :-1].clone()

        # masking target users
        item_idx_batch = torch.arange(item_idx.size(0))
        item_interaction_slice[item_idx_batch, user_idx] = 0

        # projection
        item_proj_slice = self.proj_i(item_interaction_slice.float())

        # representation learning
        item_rep_slice = self.rep_i(item_proj_slice)

        return item_rep_slice

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.n_items,
            out_features=self.hidden[0],
            bias=False,
        )
        self.proj_u = nn.Linear(**kwargs)

        kwargs = dict(
            in_features=self.n_users,
            out_features=self.hidden[0],
            bias=False,
        )
        self.proj_i = nn.Linear(**kwargs)

        components = list(self._yield_layers(self.hidden))
        self.rep_u = nn.Sequential(*components)

        components = list(self._yield_layers(self.hidden))
        self.rep_i = nn.Sequential(*components)

    def _yield_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[-1] == self.n_factors)
        ERROR_MESSAGE = f"Last representation function layer must match input size: {self.n_factors}"
        assert CONDITION, ERROR_MESSAGE