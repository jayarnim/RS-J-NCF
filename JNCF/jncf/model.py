import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden_rl: list,
        hidden_ml: list,
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
        self.hidden_rl = hidden_rl
        self.hidden_ml = hidden_ml
        self.dropout = dropout
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector = self.ml(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def ml(self, user_idx, item_idx):
        rep_user, rep_item = self.rl(user_idx, item_idx)

        concat = torch.cat(
            tensors=(rep_user, rep_item), 
            dim=-1
        )
        pred_vector = self.mlp_layers(concat)

        return pred_vector        

    def rl(self, user_idx, item_idx):
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
        self.proj_u = nn.Linear(
            in_features=self.n_items,
            out_features=self.hidden[0],
            bias=False,
        )
        self.proj_i = nn.Linear(
            in_features=self.n_users,
            out_features=self.hidden[0],
            bias=False,
        )
        self.mlp_u = nn.Sequential(
            *list(self._generate_layers(self.hidden_rl))
        )
        self.mlp_i = nn.Sequential(
            *list(self._generate_layers(self.hidden_rl))
        )
        self.mlp_layers = nn.Sequential(
            *list(self._generate_layers(self.hidden_ml))
        )
        self.logit_layer = nn.Linear(
            in_features=self.hidden[-1],
            out_features=1,
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
        CONDITION = (self.hidden_rl[-1] == self.n_factors)
        ERROR_MESSAGE = f"Last representation function layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (self.hidden_ml[0] == self.n_factors * 2)
        ERROR_MESSAGE = f"First matching function layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE