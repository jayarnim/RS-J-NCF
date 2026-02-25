import torch
import torch.nn as nn
from .components.embedding import HistoryEmbedding
from .components.representation import RepresentationLayer
from .components.matching import NeuralCollaborativeFiltering
from .components.scorer import LinearProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        interactions: torch.Tensor, 
        num_users: int,
        num_items: int,
        projection_dim: int,
        hidden_dim_rl: list,
        hidden_dim_ml: list,
        dropout: float,
    ):
        """
        Joint neural collaborative filtering for recommender systems (Chen et al., 2019)
        -----
        Implements the base structure of Joint neural collaborative filtering (J-NCF),
        MLP & history embedding based latent factor model.

        Args:
            interactions (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
            num_users (int): 
                total number of users in the dataset, U.
            num_items (int): 
                total number of items in the dataset, I.
            projection_dim (int): 
                dimensionality of user and item projection vectors.
            hidden_dim_rl (list): 
                layer dimensions for the representation. 
                (e.g., [128, 64, 32])
            hidden_dim_ml (list): 
                layer dimensions for the matching function. 
                (e.g., [64, 32, 16])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.projection_dim = projection_dim
        self.hidden_dim_rl = hidden_dim_rl
        self.hidden_dim_ml = hidden_dim_ml
        self.dropout = dropout

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        user_rep, item_rep = self.representation(user_emb, item_emb)
        predictive_vec = self.matching(user_rep, item_rep)
        return predictive_vec

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            interactions=self.interactions, 
            num_users=self.num_users,
            num_items=self.num_items,
            projection_dim=self.projection_dim,
        )
        self.embedding = HistoryEmbedding(**kwargs)

        kwargs = dict(
            input_dim=self.projection_dim,
            hidden_dim=self.hidden_dim_rl,
            dropout=self.dropout,
        )
        self.representation = RepresentationLayer(**kwargs)

        kwargs = dict(
            input_dim=self.hidden_dim_rl[-1]*2,
            hidden_dim=self.hidden_dim_ml,
            dropout=self.dropout,
        )
        self.matching = NeuralCollaborativeFiltering(**kwargs)

        kwargs = dict(
            input_dim=self.hidden_dim_ml[-1],
        )
        self.scorer = LinearProjectionLayer(**kwargs)