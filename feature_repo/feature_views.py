from datetime import timedelta

from feast import (
    FeatureView,
    Field,
)
from feast.types import Float32

from data_sources import interactions_source, items_source, users_source, item_embed_push_source, user_embed_push_source, user_items_push_source
from entities import user_entity, item_entity
from feast.types import Float32, Int32, Int64, String, Bool, Array


user_feature_view = FeatureView(
    name="user_features",
    entities=[user_entity],
    ttl=timedelta(days=365 * 6),
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="age", dtype=Int32),
        Field(name="gender", dtype=String),
        Field(name="preferences", dtype=String),
    ],
    source=users_source,
    online=False
)

item_feature_view = FeatureView(
    name="item_features",
    entities=[item_entity],
    ttl=timedelta(days=365 * 5), 
    schema=[
        Field(name="item_id", dtype=Int64),
        Field(name="category", dtype=String),
        Field(name="subcategory", dtype=String),
        Field(name="price", dtype=Float32),
        Field(name="avg_rating", dtype=Float32),
        Field(name="num_ratings", dtype=Int32),
        Field(name="popular", dtype=Bool),
        Field(name="new_arrival", dtype=Bool),
        Field(name="on_sale", dtype=Bool),
    ],
    source=items_source,
    online=False
)

interaction_feature_view = FeatureView(
    name="interactions_features",
    entities=[user_entity, item_entity],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="item_id", dtype=Int64),
        Field(name="interaction_type", dtype=String),
        Field(name="rating", dtype=Int32),
        Field(name="quantity", dtype=Int32),
    ],
    source=interactions_source,
    online=False
)

item_embedding_view = FeatureView(
    name="item_embedding",
    entities=[item_entity],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="item_id", dtype=Int64),
        Field(
            name="embedding",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="cosine",
        ),
    ],
    source=item_embed_push_source,
    online=True
)

user_embedding_view = FeatureView(
    name="user_embedding",
    entities=[user_entity],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(
            name="embedding",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="cosine",
        )
    ],
    source=user_embed_push_source,
    online=True
)

user_items_view = FeatureView(
    name="user_items",
    entities=[user_entity],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name='top_k_item_ids', dtype=Array(Int64), vector_index=False)
    ],
    source=user_items_push_source,
    online=True
)