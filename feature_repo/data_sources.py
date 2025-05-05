from feast import FileSource, PushSource
from feast.data_format import ParquetFormat
import os
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)

feast_path = 'feature_repo'
data_path = 'data'

users_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_users.parquet'),
    timestamp_field="signup_date",
)
interactions_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_interactions.parquet'),
    timestamp_field="timestamp",
)
neg_interactions_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_neg_interactions.parquet'),
    timestamp_field="timestamp",
)
items_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_items.parquet'),
    timestamp_field="arrival_date",
)
items_embed_dummy_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'dummy_item_embed.parquet'),
    timestamp_field="event_timestamp",
)
users_embed_dummy_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'dummy_user_embed.parquet'),
    timestamp_field="event_timestamp",
)
users_items_dummy_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'user_items.parquet'),
    timestamp_field="event_timestamp",
)

item_embed_push_source = PushSource(
    name='item_embed_push_source',
    batch_source=items_embed_dummy_source
)

user_embed_push_source = PushSource(
    name='user_embed_push_source',
    batch_source=users_embed_dummy_source    
)

user_items_push_source = PushSource(
    name='user_items_push_source',
    batch_source=users_items_dummy_source    
)

# interaction_stream_source = KafkaSource(
#     name="intercation_stream_source",
#     kafka_bootstrap_servers='xray-cluster-kafka-bootstrap.jary-feast-example.svc.cluster.local:9092',
#     topic="interactions",
#     timestamp_field="timestamp",
#     batch_source=interactions_source,
#     message_format=JsonFormat(
#         schema_json="user_id integer, item_id integer, timestamp timestamp, interaction_type string, rating integer, quantity integer"
#     ),
#     watermark_delay_threshold=timedelta(minutes=5),
# )


interaction_stream_source = PostgreSQLSource(
    name="interaction_stream_source",
    query="SELECT * FROM stream_interaction",
    # timestamp_field="event_timestamp",
    timestamp_field="timestamp",
    # created_timestamp_column="created",
)