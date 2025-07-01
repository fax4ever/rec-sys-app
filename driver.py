from datetime import datetime

import json
import torch
from feast import FeatureStore
from feast.data_source import PushMode

from models.train_two_tower import create_and_train_two_tower
from models.data_util import data_preproccess

from service.search_by_text import SearchService
from service.search_by_image import SearchByImageService
from service.clip_encoder import ClipEncoder
from service.dataset_provider import LocalDatasetProvider

"""
Code copied and adapted from https://github.com/RHEcosystemAppEng/rec-sys-workflow/blob/main/train-workflow.py
"""


def main():
    store = FeatureStore(repo_path="feature_repo/")
    store.refresh_registry()
    print('registry refreshed')

    dataset_provider = LocalDatasetProvider(store)

    # retrieve datasets for training
    item_df = dataset_provider.item_df()
    user_df = dataset_provider.user_df()
    interaction_df = dataset_provider.interaction_df()

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
