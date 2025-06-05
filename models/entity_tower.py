import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict


class EntityTower(nn.Module):
    def __init__(self, num_numerical: int, num_of_categories: int, d_model: int=64, text_embed_dim: int = 384, image_embed_dim: int = 384, dim_ratio: Dict = {'numeric': 1, 'categorical': 2, 'text': 7, 'image': 0}):
        super().__init__()
        # Calculate ratio weight for dimension allocation
        ratio_weight = d_model / sum(dim_ratio.values())
        
        # Assign dimensions based on ratios
        numerical_dim = int(dim_ratio['numeric'] * ratio_weight)
        categorical_dim = int(dim_ratio['categorical'] * ratio_weight)
        self.image_dim = int(dim_ratio['image'] * ratio_weight)
        self.text_dim = d_model - num_numerical - categorical_dim - self.image_dim
        
        # Create embedding modules for categorical features.
        self.categorical_embed = nn.Embedding(num_of_categories, categorical_dim)

        # Create projection and normalization modules for each numeric feature.
        self.numeric_norm = nn.BatchNorm1d(num_numerical)
        self.numeric_embed = nn.Linear(num_numerical, numerical_dim)
        
        # prject text to lower dimension
        self.project_text = nn.Linear(text_embed_dim, self.text_dim)
        
        # prject image to lower dimension
        if self.image_dim > 0:
            self.project_image = nn.Linear(image_embed_dim, self.image_dim)
        
        # Simple linear model with relu can be replace with transformer encoder
        self.fn1 = nn.Linear(d_model, d_model * 2)
        self.fn2 = nn.Linear(d_model * 2, d_model)
        
        self.norm = nn.RMSNorm(d_model)
        self.norm1 = nn.RMSNorm(d_model)
        
        self.relu = nn.ReLU()

    # def forward(self, x_numeric: Tensor, x_categorical: Tensor):
    def forward(self, numerical_features: Tensor, categorical_features: Tensor, text_features: Tensor, url_image: Tensor):
        """
        Process input features for a machine learning model, handling numerical, categorical, text, and image data.

        Args:
            numerical_features (Tensor): Numerical data with        shape: (n_samples, n_col)
            categorical_features (Tensor): Categorical data with    shape: (n_samples, n_col).
            text_features (Tensor): Text data embeddings with       shape: (n_samples, n_col, dim).
            url_image (Tensor): Image data (e.g., from URLs) with   shape: (n_samples, n_col).
        """

        # Process numeric.
        x_numeric = self.numeric_embed(self.numeric_norm(numerical_features)) # n_samples, numeric_dim
        # Process categorical.
        x_categorical = self.categorical_embed(categorical_features)          # n_samples, cat_dim
        # Process textual.
        x_text = self.project_text(text_features).mean(dim=1) # n_samples, text_dim
        # Proccess image
        # if self.image_dim != 0:
        #     pass # TODO implement mt later
        #     # x_image = self.project_image()
            
        # Concatenate all feature representations.
        
        x = torch.cat([x_numeric, x_categorical, x_text], dim=-1)
        
        x = self.norm(x)
        y = self.relu(self.fn1(self.norm1(x)))
        x = self.relu(self.fn2(y)) + x
        return x
        
