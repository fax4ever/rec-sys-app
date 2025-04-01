import torch
import torch.nn as nn
from torch import Tensor

class UserTower(nn.Module):
    def __init__(self, d_model: int=64, preferences_category_count: int=5, gender_category_count: int=3):
        super().__init__()
        feature_dim_ratio = {
            'age': 2,
            'gender': 2,
            'signup_date': 1,
            'preferences': 5
        }
        total_ratio = sum(feature_dim_ratio.values())
        dim_reminder = d_model % total_ratio
        unit_dim = d_model // total_ratio
        
        # dimention size calculation for each feature
        age_dim = feature_dim_ratio['age'] * unit_dim
        gender_dim = feature_dim_ratio['gender'] * unit_dim
        signup_date_dim = feature_dim_ratio['signup_date'] * unit_dim
        preferences_dim = feature_dim_ratio['preferences'] * unit_dim + dim_reminder
        
        # Embedding categorical features
        self.gender_embed = nn.Embedding(gender_category_count, gender_dim)
        self.preferences_embed = nn.Embedding(preferences_category_count, preferences_dim)
        
        # Encoding numeric features
        self.age_encoder = nn.Linear(1, age_dim)
        self.age_norm = nn.RMSNorm(age_dim)
        self.signup_date_encoder = nn.Linear(1, signup_date_dim)
        self.signup_date_norm = nn.RMSNorm(signup_date_dim)
        
        
    def forward(self, age: Tensor, gender: Tensor, signup_date: Tensor, preferences: Tensor):
        # project numerical features
        age = self.age_norm(self.age_encoder(age))
        signup_date = self.signup_date_norm(self.signup_date_encoder(signup_date))
        
        # embed categorical features
        gender = self.gender_embed(gender)
        preferences = self.preferences_embed(preferences)
        
        return torch.cat((age, gender, signup_date, preferences), dim=-1)