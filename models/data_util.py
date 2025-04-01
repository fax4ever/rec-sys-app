import pandas as pd
import torch
from typing import Dict, List
from pandas.api.types import is_datetime64_any_dtype

def data_preproccess(df: pd.DataFrame):
    category_values = {
        'preferences': ['Electronics', 'Books', 'Clothing', 'Home', 'Sports'],
        'category': ['Electronics', 'Books', 'Clothing', 'Home', 'Sports'],
        'subcategory': ['Smartphones', 'Laptops', 'Cameras', 'Audio', 'Accessories', 'Fiction', 'Non-fiction', 'Science', 'History', 'Self-help', 'Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories', 'Kitchen', 'Furniture', 'Decor', 'Bedding', 'Appliances', 'Fitness', 'Outdoor', 'Team Sports', 'Footwear', 'Equipment'],
        'interaction_type': ['view', 'cart', 'purchase', 'rate'],
        'gender':['M', 'F', 'Other']
        
    }
    if 'category' in df.columns:
        df.rename(columns={'event_timestamp': 'arrival_date'}, inplace=True)
    elif 'gender' in df.columns:
        df.rename(columns={'event_timestamp': 'signup_date'}, inplace=True)
    elif 'interaction_type' in df.columns:
        df.rename(columns={'event_timestamp': 'timestamp'}, inplace=True)
        
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
        x_feature = torch.tensor(x_feature)
        if not is_category:
            x_feature = x_feature.view(-1, 1).to(torch.float32)
        proccesed_tensor_dict[feature] = x_feature
    return proccesed_tensor_dict
            
            