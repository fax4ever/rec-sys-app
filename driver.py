from datetime import datetime
from pprint import pprint

import json
import pandas as pd
import torch
from feast import FeatureStore
from feast.data_source import PushMode

from models.train_two_tower import create_and_train_two_tower
from models.data_util import data_preproccess

from service.search_by_text import SearchService
from service.search_by_image import SearchByImageService
from service.clip_encoder import ClipEncoder

"""
Code copied and adapted from https://github.com/RHEcosystemAppEng/rec-sys-workflow/blob/main/train-workflow.py
"""


def main():
    store = FeatureStore(repo_path="feature_repo/")
    store.refresh_registry()
    print('registry refreshed')

    # load feature services
    item_service = store.get_feature_service("item_service")
    user_service = store.get_feature_service("user_service")
    interaction_service = store.get_feature_service("interaction_service")
    print('service loaded')

    users_ids = pd.read_parquet('./feature_repo/data/recommendation_interactions.parquet')
    pprint(users_ids)
    user_ids = users_ids['user_id'].unique().tolist()
    item_ids = users_ids['item_id'].unique().tolist()

    # select which items to use for the training
    item_entity_df = pd.DataFrame.from_dict(
        {
            'item_id': item_ids,
            'event_timestamp': [datetime(2025, 1, 1)] * len(item_ids)
        }
    )
    # select which users to use for the training
    user_entity_df = pd.DataFrame.from_dict(
        {
            'user_id': user_ids,
            'event_timestamp': [datetime(2025, 1, 1)] * len(user_ids)
        }
    )
    # Select which item-user interactions to use for the training
    item_user_interactions_df = pd.read_parquet('./feature_repo/data/interactions_item_user_ids.parquet')
    item_user_interactions_df['event_timestamp'] = datetime(2025, 1, 1)

    # retrieve datasets for training
    item_df = store.get_historical_features(entity_df=item_entity_df, features=item_service).to_df()
    user_df = store.get_historical_features(entity_df=user_entity_df, features=user_service).to_df()
    interaction_df = store.get_historical_features(entity_df=item_user_interactions_df, features=interaction_service).to_df()

    item_df.to_parquet("./feature_repo/data/item_df_output.parquet")
    user_df.to_parquet("./feature_repo/data/user_df_output.parquet")
    interaction_df.to_parquet("./feature_repo/data/interaction_df_output.parquet")

    item_encoder, user_encoder, models_definition = create_and_train_two_tower(item_df, user_df, interaction_df,
                                                                               return_model_definition=True)

    torch.save(item_encoder.state_dict(), "params/item_encoder.dat")
    torch.save(user_encoder.state_dict(), "params/user_encoder.dat")
    with open("params/models_definition.dat", 'w') as f:
        json.dump(models_definition, f)

    # Inference
    item_encoder.eval()
    user_encoder.eval()

    # Create a new table to be push to the online store
    item_embed_df = item_df[['item_id']].copy()
    user_embed_df = user_df[['user_id']].copy()

    proccessed_items = data_preproccess(item_df)
    proccessed_users = data_preproccess(user_df)

    # Encode the items and users
    item_embed_df['embedding'] = item_encoder(**proccessed_items).detach().numpy().tolist()
    user_embed_df['embedding'] = user_encoder(**proccessed_users).detach().numpy().tolist()

    # Add the current timestamp
    item_embed_df['event_timestamp'] = datetime.now()
    user_embed_df['event_timestamp'] = datetime.now()

    # Push the new embedding to the offline and online store
    store.push('item_embed_push_source', item_embed_df, to=PushMode.ONLINE, allow_registry_cache=False)
    store.push('user_embed_push_source', user_embed_df, to=PushMode.ONLINE, allow_registry_cache=False)

    # Store the embedding of text features for search by text
    item_text_features_embed = item_df[['item_id']].copy()
    item_text_features_embed['about_product_embedding'] = proccessed_items['text_features'].detach()[:, 1, :].numpy().tolist()
    item_text_features_embed['event_timestamp'] = datetime.now()

    store.push('item_textual_features_embed', item_text_features_embed, to=PushMode.ONLINE, allow_registry_cache=False)

    # Store the embedding of clip features for search by image
    clip_encoder = ClipEncoder()
    item_clip_features_embed = clip_encoder.clip_embeddings(item_df)
    store.push('item_clip_features_embed', item_clip_features_embed, to=PushMode.ONLINE, allow_registry_cache=False)

    # Materialize the online store
    store.materialize_incremental(datetime.now(), feature_views=['item_embedding', 'user_items', 'item_features', 'item_textual_features_embed'])

    # Calculate user recommendations for each user
    item_embedding_view = 'item_embedding'
    k = 64
    item_recommendation = []
    for user_embed in user_embed_df['embedding']:
        item_recommendation.append(
            store.retrieve_online_documents(
                query=user_embed,
                top_k=k,
                features=[f'{item_embedding_view}:item_id']
            ).to_df()['item_id'].to_list()
        )

    # Pushing the calculated items to the online store
    user_items_df = user_embed_df[['user_id']].copy()
    user_items_df['event_timestamp'] = datetime.now()
    user_items_df['top_k_item_ids'] = item_recommendation

    store.push('user_items_push_source', user_items_df, to=PushMode.ONLINE, allow_registry_cache=False)

    search_service = SearchService(store)
    items = search_service.search_by_text("excellent value for money", 10)
    print(items)

    search_by_image_service = SearchByImageService(store, clip_encoder)
    items = search_by_image_service.search_by_image_link("http://images.cocodataset.org/val2017/000000039769.jpg", 10)
    print(items)

if __name__ == '__main__':
    main()
