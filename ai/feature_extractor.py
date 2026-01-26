"""
Custom Feature Extractor pour SB3 avec embeddings spatiaux.

Ce module implémente un extracteur hybride qui combine:
- Des embeddings pour les positions spatiales (balle, raquette)
- Des embeddings pour l'angle de la raquette
- Des features continues (vitesses, distances, flags)

Compatible avec MultiInputPolicy de Stable-Baselines3.
"""

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HybridFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor hybride pour le Ping-Pong.
    
    Architecture:
    1. Embeddings spatiaux 16x16 pour la position de la balle (256 cellules → embed_dim)
    2. Embeddings spatiaux 16x16 pour la position de la raquette (256 cellules → embed_dim)
    3. Embeddings pour l'angle de la raquette (16 bins → embed_dim)
    4. Features continues (14 valeurs) directement concaténées
    
    Sortie: Un vecteur de taille (3 * embed_dim + 14) qui sera utilisé par les 
            couches fully-connected de la policy et value networks de SB3.
    
    Args:
        observation_space: L'espace d'observation (Dict space de gymnasium)
        embed_dim: Dimension des embeddings (par défaut 16)
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, embed_dim: int = 16):
        # Calculer la dimension totale de sortie
        # 3 embeddings (ball, paddle, angle) * embed_dim + 14 features continues
        features_dim = (3 * embed_dim) + 14
        
        super().__init__(observation_space, features_dim)
        
        # === EMBEDDINGS ===
        # Grille spatiale 16x16 = 256 cellules
        grid_size = 16
        num_cells = grid_size * grid_size  # 256
        
        # Embedding pour la position de la balle
        self.ball_embedding = nn.Embedding(num_cells, embed_dim)
        
        # Embedding pour la position de la raquette
        self.paddle_embedding = nn.Embedding(num_cells, embed_dim)
        
        # Embedding pour l'angle de la raquette (16 bins cycliques)
        self.angle_embedding = nn.Embedding(16, embed_dim)
        
        self.embed_dim = embed_dim
        
    def forward(self, observations: dict) -> th.Tensor:
        """
        Extrait les features à partir des observations.
        
        Args:
            observations: Dict contenant:
                - 'ball_idx': Tensor de shape (batch, 1) avec index de cellule balle
                - 'paddle_idx': Tensor de shape (batch, 1) avec index de cellule raquette
                - 'angle_idx': Tensor de shape (batch, 1) avec index d'angle
                - 'continuous': Tensor de shape (batch, 14) avec features continues
        
        Returns:
            Tensor de shape (batch, features_dim) avec toutes les features concaténées
        """
        # 1. Récupérer les index et s'assurer qu'ils sont au bon format (long)
        # Les embeddings PyTorch nécessitent des indices de type long
        ball_idx = observations["ball_idx"].long().squeeze(-1)  # (batch,)
        paddle_idx = observations["paddle_idx"].long().squeeze(-1)  # (batch,)
        angle_idx = observations["angle_idx"].long().squeeze(-1)  # (batch,)
        
        # 2. Passer dans les embeddings
        ball_embed = self.ball_embedding(ball_idx)  # (batch, embed_dim)
        paddle_embed = self.paddle_embedding(paddle_idx)  # (batch, embed_dim)
        angle_embed = self.angle_embedding(angle_idx)  # (batch, embed_dim)
        
        # 3. Récupérer les features continues
        continuous = observations["continuous"]  # (batch, 14)
        
        # 4. Concaténer tout le long de la dimension des features
        # Résultat: (batch, 3*embed_dim + 14)
        return th.cat([ball_embed, paddle_embed, angle_embed, continuous], dim=1)
