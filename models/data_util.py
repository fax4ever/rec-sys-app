import pandas as pd
import torch
from typing import Dict, List
from pandas.api.types import is_datetime64_any_dtype
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class
class UserItemMagnitudeDataset(Dataset):
    def __init__(self, items, users, magnitude):
        """
        Args:
            items (dict): Dictionary with keys mapping to tensors of shape (n_samples, 1)
            users (dict): Dictionary with keys mapping to tensors of shape (n_samples, 1)
            magnitude (tensor): Tensor of shape (n_samples,)
        """
        self.items = items
        self.users = users
        self.magnitude = magnitude
        
        # Verify that all tensors have consistent number of samples
        n_samples = len(magnitude)
        for user_tensor in users.values():
            assert user_tensor.shape[0] == n_samples, "User tensor size mismatch"
        for item_tensor in items.values():
            assert item_tensor.shape[0] == n_samples, "Item tensor size mismatch"
            
    def __len__(self):
        """Returns the total number of samples"""
        return len(self.magnitude)
    
    def __getitem__(self, idx):
        """Returns a single sample"""
        
        # Get item data for this index
        item_sample = {key: tensor[idx] for key, tensor in self.items.items()}
        
        # Get user data for this index
        user_sample = {key: tensor[idx] for key, tensor in self.users.items()}
        
        # Get magnitude for this index
        magnitude_sample = self.magnitude[idx]
        
        return item_sample, user_sample, magnitude_sample

def data_preproccess(df: pd.DataFrame):
    category_values = {
        'preferences': ['Electronics', 'Books', 'Clothing', 'Home', 'Sports'],
        'category': ['Electronics', 'Books', 'Clothing', 'Home', 'Sports'],
        'subcategory': ['Smartphones', 'Laptops', 'Cameras', 'Audio', 'Accessories', 'Fiction', 'Non-fiction', 'Science', 'History', 'Self-help', 'Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories', 'Kitchen', 'Furniture', 'Decor', 'Bedding', 'Appliances', 'Fitness', 'Outdoor', 'Team Sports', 'Footwear', 'Equipment'],
        # 'interaction_type': ['view', 'cart', 'purchase', 'rate'],
        'gender':['M', 'F', 'Other']
        
    }
    if 'category' in df.columns:
        df.rename(columns={'event_timestamp': 'arrival_date'}, inplace=True)
    elif 'gender' in df.columns:
        df.rename(columns={'event_timestamp': 'signup_date'}, inplace=True)
    # elif 'interaction_type' in df.columns:
    #     df.rename(columns={'event_timestamp': 'timestamp'}, inplace=True)
        
    features = [feature for feature in df.columns if not feature.endswith('_timestamp') and not feature.endswith('_id')]
    proccesed_tensor_dict = dict()
    for feature in features:
        # feature is category 
        is_category = feature in category_values.keys()
        if is_category:
            categories = category_values.get(feature)
            # map numbers to each category
            category_num = {category: i for category, i, in zip(categories, range(len(categories)))}
            x_feature = df[feature].map(category_num)
        # datetime case
        elif is_datetime64_any_dtype(df[feature]):
            x_feature = df[feature].apply(lambda x: x.toordinal())
        # numerical case
        else:
            x_feature = df[feature]
            
        # parse to tensor
        x_feature = torch.tensor(x_feature.values)
        if not is_category:
            x_feature = x_feature.view(-1, 1).to(torch.float32)
        proccesed_tensor_dict[feature] = x_feature
    return proccesed_tensor_dict

def preproccess_pipeline(item_df: pd.DataFrame, user_df: pd.DataFrame, interaction_df_pos: pd.DataFrame, interaction_df_neg: pd.DataFrame):
    item_df_pos, user_df_pos, inter_df_pos = _reorder(item_df, user_df, interaction_df_pos)
    item_df_neg, user_df_neg, inter_df_neg = _reorder(item_df, user_df, interaction_df_neg)
    magnitude_pos = _calculate_interaction_loss(inter_df_pos, is_positive=True)
    magnitude_neg = _calculate_interaction_loss(inter_df_neg, is_positive=False)
    
    item_df = pd.concat([item_df_pos, item_df_neg], axis=0)
    user_df = pd.concat([user_df_pos, user_df_neg], axis=0)
    magnitude = torch.Tensor(pd.concat([magnitude_pos, magnitude_neg], axis=0).values)
    
    item_dict = data_preproccess(item_df)
    user_dict = data_preproccess(user_df)
    
    return UserItemMagnitudeDataset(item_dict, user_dict, magnitude)

def _reorder(item_df: pd.DataFrame, user_df: pd.DataFrame, interaction_df: pd.DataFrame):
    merged_df = (
        interaction_df
        .merge(item_df, on='item_id')
        .merge(user_df, on='user_id')
    )
    return merged_df[item_df.columns], merged_df[user_df.columns], merged_df[interaction_df.columns]

def _calculate_interaction_loss(inter_df: pd.DataFrame, is_positive: bool, a: float=1.1, magnitude_defualt: float=11.265591558187197):
    # 'interaction_type': ['view', 'cart', 'purchase', 'rate'],
    # 'rating': 1-5, or non
    # 'quantity': 1-3 or non
    # ['view', 'cart', 'purchase', 'rate']
    panisment = {
        'interaction_type': { # view and click vs view and not click
            'view': lambda x: x / a if is_positive else x * a, 
            'cart': lambda x: x / (a * 3), 
            'purchase': lambda x: x / (a * 10),
            'rate': lambda x: x 
        },
        'rating': {
            1.0: lambda x: x * (a * 2),
            2.0: lambda x: x * (a * 1),
            3.0: lambda x: x,
            4.0: lambda x: x / (a * 1),
            5.0: lambda x: x / (a * 2),
            -1.0: lambda x: x # placeholder
        },
        'quantity': {
            1.0: lambda x: x,
            2.0: lambda x: x / (a * 1),
            3.0: lambda x: x / (a * 2),
            -1.0: lambda x: x # placeholder
        }
    }
    inter_df.fillna(-1.0, inplace=True)
    inter_df['magnitude'] = magnitude_defualt
    for col in [punishment_col for punishment_col in panisment.keys() if punishment_col in inter_df.columns]:
        inter_df[col] = inter_df[col].map(panisment[col])
        inter_df['magnitude'] = inter_df.apply(lambda row: row[col](row['magnitude']), axis=1)
            
    return inter_df['magnitude']