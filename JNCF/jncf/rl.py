import torch
import torch.nn as nn


class RepresentationFunction(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        super(RepresentationFunction, self).__init__()
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
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        rep_user = self.user(user_idx, item_idx)
        rep_item = self.item(user_idx, item_idx)
        return rep_user, rep_item

    def user(self, user_idx, item_idx):
        # get user vector from interactions
        user_slice = self.interactions[user_idx, :-1].clone()

        # masking target items
        user_batch = torch.arange(user_idx.size(0))
        user_slice[user_batch, item_idx] = 0

        # projection
        proj_user = self.proj_u(user_slice.float())

        # representation learning
        rep_user = self.mlp_u(proj_user)

        return rep_user

    def item(self, user_idx, item_idx):
        # get item vector from interactions
        item_slice = self.interactions.T[item_idx, :-1].clone()

        # masking target users
        item_batch = torch.arange(item_idx.size(0))
        item_slice[item_batch, user_idx] = 0

        # projection
        proj_item = self.proj_i(item_slice.float())

        # representation learning
        rep_item = self.mlp_i(proj_item)

        return rep_item

    def _init_layers(self):
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

        self.mlp_u = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )

        self.mlp_i = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )

    def _generate_layers(self, hidden):
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