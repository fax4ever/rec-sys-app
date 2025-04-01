import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

# Set random seed for reproducibility
np.random.seed(42)

# Generate user data
def generate_users(num_users, from_id = 0):
    users = []
    for user_id in range(1 + from_id, num_users + from_id + 1):
        age = np.random.randint(18, 65)
        gender = np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04])
        signup_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
        
        # Generate user preferences (categories they tend to like)
        preferences = np.random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Sports'])
        
        users.append({
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'signup_date': signup_date,
            'preferences': preferences
        })
    
    return pd.DataFrame(users)

# Generate item data
def generate_items(num_items):
    items = []
    categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports']
    subcategories = {
        'Electronics': ['Smartphones', 'Laptops', 'Cameras', 'Audio', 'Accessories'],
        'Books': ['Fiction', 'Non-fiction', 'Science', 'History', 'Self-help'],
        'Clothing': ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories'],
        'Home': ['Kitchen', 'Furniture', 'Decor', 'Bedding', 'Appliances'],
        'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Footwear', 'Equipment']
    }
    
    for item_id in range(1, num_items + 1):
        category = np.random.choice(categories)
        subcategory = np.random.choice(subcategories[category])
        price = np.round(np.random.uniform(5, 500), 2)
        avg_rating = np.round(np.random.uniform(1, 5), 1)
        num_ratings = np.random.randint(0, 1000)
        
        # Item features that could be useful for recommendation
        features = {
            'popular': np.random.random() > 0.7,
            'new_arrival': np.random.random() > 0.8,
            'on_sale': np.random.random() > 0.75
        }
        arrival_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365), 
                                                      hours=np.random.randint(0, 24),
                                                      minutes=np.random.randint(0, 60))
        items.append({
            'item_id': item_id,
            'category': category,
            'subcategory': subcategory,
            'price': price,
            'avg_rating': avg_rating,
            'num_ratings': num_ratings,
            'popular': features['popular'],
            'new_arrival': features['new_arrival'],
            'on_sale': features['on_sale'],
            'arrival_date': arrival_date
        })
    return pd.DataFrame(items)

# Generate interactions between users and items
def generate_interactions(users_df: pd.DataFrame, items_df: pd.DataFrame, num_interactions: int):
    interactions = []
    
    # Ensure we have sufficient users and items
    num_users = len(users_df)
    num_items = len(items_df)
    
    for _ in range(num_interactions):
        user_id = np.random.randint(1, num_users + 1)
        
        # Users are more likely to interact with items in their preferred categories
        user_prefs = users_df.loc[users_df['user_id'] == user_id, 'preferences'].iloc[0].split(',')
        
        # Biased item selection based on user preferences
        if np.random.random() < 0.7 and user_prefs:  # 70% chance to select from preferred categories
            preferred_items = items_df[items_df['category'].isin(user_prefs)]
            if not preferred_items.empty:
                item = preferred_items.sample(1).iloc[0]
                item_id = item['item_id']
            else:
                item_id = np.random.randint(1, num_items + 1)
        else:
            item_id = np.random.randint(1, num_items + 1)
        
        # Generate interaction details
        timestamp = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365), 
                                                      hours=np.random.randint(0, 24),
                                                      minutes=np.random.randint(0, 60))
        
        # Different types of interactions
        interaction_type = np.random.choice(['view', 'cart', 'purchase', 'rate'], p=[0.6, 0.2, 0.15, 0.05])
        
        # Additional data based on interaction type
        if interaction_type == 'rate':
            rating = np.random.randint(1, 6)  # 1-5 rating
        else:
            rating = None
            
        if interaction_type == 'purchase':
            quantity = np.random.randint(1, 4)
        else:
            quantity = None
        
        interactions.append({
            'interaction_id': len(interactions) + 1,
            'user_id': user_id,
            'item_id': item_id,
            'timestamp': timestamp,
            'interaction_type': interaction_type,
            'rating': rating,
            'quantity': quantity
        })
    
    return pd.DataFrame(interactions)

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Generate Recommendation System dataset')
    parser.add_argument('--n_users', help='Number of users', type=int, default=1000)
    parser.add_argument('--n_items', type=int, help='Number of items', default=5000)
    parser.add_argument('--n_interactions', help='Number of interactions of users and items', default=20000, type=int)

    args = parser.parse_args()
    # Generate the datasets
    users = generate_users(args.n_users)
    items = generate_items(args.n_items)
    interactions = generate_interactions(users, items, args.n_interactions)

    # Display sample of each dataset
    print("Users sample:")
    print(users.head())
    print("\nItems sample:")
    print(items.head())
    print("\nInteractions sample:")
    print(interactions.head())

    # Save to parquet files
    users.to_parquet('feature_repo/data/recommendation_users.parquet', index=False)
    items.to_parquet('feature_repo/data/recommendation_items.parquet', index=False)
    interactions.to_parquet('feature_repo/data/recommendation_interactions.parquet', index=False)
    interactions[['item_id', 'user_id']].to_parquet('feature_repo/data/interactions_item_user_ids.parquet', index=False)
    
    k = 10
    
    # Create dummy dataframes for push source
    dummy_item_embed_df = pd.DataFrame(columns=['item_id', 'embedding', 'event_timestamp'], data=[[1, [1.,2.], datetime.now() + timedelta(days=365)]]) # used for type casting will be removed automaticly
    dummy_user_items_df = pd.DataFrame(columns=['user_id', 'top_k_item_ids', 'event_timestamp'], data=[[1, [1, 2], datetime.now() + timedelta(days=365)]]) # used for type casting will be removed automaticly
    # dummy_user_embed_df = pd.DataFrame(columns=['user_id', 'embedding', 'event_timestamp', 'top_k_items'], data=[[1, [1.,2.], datetime.now() + timedelta(days=365), list(range(k))]]) # used for type casting will be removed automaticly
    dummy_user_embed_df = pd.DataFrame(columns=['user_id', 'embedding', 'event_timestamp'], data=[[1, [1.,2.], datetime.now() + timedelta(days=365)]]) # used for type casting will be removed automaticly
    
    # dummy_item_embed_df = dummy_item_embed_df.astype({'item_id': 'int64', 'event_timestamp': 'datetime64[us]', 'embedding': 'object'})
    # dummy_user_embed_df = dummy_user_embed_df.astype({'user_id': 'int64', 'event_timestamp': 'datetime64[us]', 'embedding': 'object'})
    
    dummy_item_embed_df.to_parquet('feature_repo/data/dummy_item_embed.parquet', index=False)
    dummy_user_embed_df.to_parquet('feature_repo/data/dummy_user_embed.parquet', index=False)
    dummy_user_items_df.to_parquet('feature_repo/data/user_items.parquet', index=False)
    
