import torch
import torch.nn as nn
from . import rl, ml


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
        """
        Joint neural collaborative filtering for recommender systems (Chen et al., 2019)
        -----
        Implements the base structure of Joint neural collaborative filtering (J-NCF),
        MLP & history embedding based latent factor model.

        Args:
            n_users (int): 
                total number of users in the dataset, U.
            n_items (int): 
                total number of items in the dataset, I.
            n_factors (int): 
                dimensionality of user and item latent representation vectors, K.
            hidden_rl (list): 
                layer dimensions for the representation. 
                (e.g., [64, 32, 16, 8])
            hidden_ml (list): 
                layer dimensions for the matching function. 
                (e.g., [64, 32, 16, 8])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
            interaction (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
        """
        super().__init__()

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
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        return self.score(user_idx, item_idx)

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])    
        """
        logit = self.score(user_idx, item_idx)
        prob = torch.sigmoid(logit)
        return prob

    def score(self, user_idx, item_idx):
        # represenation learning
        user_embed_slice, item_embed_slice = self.rl(user_idx, item_idx)
        # matching function learning
        pred_vector = self.ml(user_embed_slice, item_embed_slice)
        # predict
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden_rl,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.rl = rl.Module(**kwargs)
        
        kwargs = dict(
            n_factors=self.n_factors,
            hidden=self.hidden_ml,
            dropout=self.dropout,
        )
        self.ml = ml.Module(**kwargs)

        kwargs = dict(
            in_features=self.hidden_ml[-1],
            out_features=1,
        )
        self.pred_layer = nn.Linear(**kwargs)