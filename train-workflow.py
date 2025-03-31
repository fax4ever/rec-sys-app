from kfp import dsl, compiler
from typing import List, Dict
import os
from kfp import kubernetes

@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=["feast[postgres]>=0.46.0"],
)
def generate_candidates(out_data_path: dsl.OutputPath()):
    pass

@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311", # TODO
    packages_to_install=[""],
)
def train_model(out_data_path: dsl.OutputPath()):
    pass
    

@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311", #TODO change image to 
    packages_to_install=["feast[postgres]>=0.46.0, pandas, numpy"],
)
def load_data_from_feast(out_data_path: dsl.OutputPath()):
    from feast import FeatureStore
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    # TODO make sure the the image have updated feature_store.yaml
    store = FeatureStore()
    # load feature services
    item_service = store.get_feature_service("item_service")
    user_service = store.get_feature_service("user_service")
    interaction_service = store.get_feature_service("interaction_service")

    num_users = 1_000
    n_items = 5_000

    user_ids = list(range(1, num_users+ 1))
    item_ids = list(range(1, n_items+ 1))

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
    item_user_interactions_df = pd.read_parquet('./data/interactions_item_user_ids.parquet')
    item_user_interactions_df['event_timestamp'] = datetime(2025, 1, 1)

    # retrive datasets for training
    item_df = store.get_historical_features(entity_df=item_entity_df, features=item_service).to_df()
    user_df = store.get_historical_features(entity_df=user_entity_df, features=user_service).to_df()
    interaction_df = store.get_historical_features(entity_df=item_user_interactions_df, features=interaction_service).to_df()

    
@dsl.pipeline(name=os.path.basename(__file__).replace(".py", ""))
def rag_llm_pipeline(
    load_from_repo: bool,
    load_from_s3: bool,
    load_from_urls: bool,
    TODO_must_update_secret_ds_pipeline_config_llm_rag: str = "",
):

    load_data_task = load_data_from_repo()
    configuration_load_task(load_data_task, "REPO_CONFIG")
    split_embed_pipeline(load_data_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=rag_llm_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )