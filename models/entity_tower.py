import torch
import torch.nn as nn
from torch import Tensor


class EntityTower(nn.Module):
    def __init__(self, num_numerical: int, num_of_categories: int, numerical_dim: int, d_model: int=64):
        super().__init__()
        categorical_dim = d_model - numerical_dim
        
        # Create embedding modules for categorical features.
        self.categorical_embed = nn.Embedding(num_of_categories, categorical_dim)

        # Create projection and normalization modules for each numeric feature.
        self.numeric_norm = nn.BatchNorm1d(num_numerical)
        self.numeric_embed = nn.Linear(num_numerical, numerical_dim)
        
        self.fn1 = nn.Linear(d_model, d_model * 2)
        self.fn2 = nn.Linear(d_model * 2, d_model)
        
        self.norm = nn.RMSNorm(d_model)
        self.norm1 = nn.RMSNorm(d_model)
        
        self.relu = nn.ReLU()

    def forward(self, x_numeric: Tensor, x_categorical: Tensor):
        # Process categorical features.
        x_numeric = self.numeric_embed(self.numeric_norm(x_numeric))
        # Process numeric features using a loop.
        num_inputs = self.categorical_embed(x_categorical)
        # Concatenate all feature representations.
        x = torch.cat([x_numeric, num_inputs], dim=-1)
        
        x = self.norm(x)
        y = self.relu(self.fn1(self.norm1(x)))
        x = self.relu(self.fn2(y)) + x
        return x
        
