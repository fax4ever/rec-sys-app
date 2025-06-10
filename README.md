# Recommendation System Application
## Overview
This application implements a recommendation system using a two-tower neural network architecture, integrated with Feast for feature management. It generates synthetic datasets for users, items, and interactions, trains a model to produce user and item embeddings, and provides personalized item recommendations. 

## Features

* Data Generation: Creates synthetic datasets for users, items, positive interactions, and negative interactions using dataset_gen.py.
* Feature Store: Utilizes Feast to manage and serve features, with configurations defined in feature_repo/.
* Model Training: Trains a two-tower model (UserTower and ItemTower) to generate embeddings, implemented in models/.
* Filtering: Applies rule-based filtering (availability, demographic, history, and contextual) to refine recommendations (models/filtering.py).

## Driver

Before to start, select an environment with (transformers, pytorch, feast, ...)

```shell
conda activate rec-sys
```

To play with the application driver:

### Run Postgres image

```shell
podman run -e POSTGRES_USER=app \
           -e POSTGRES_PASSWORD=placeholder \
           -e POSTGRES_DB=app \
           --name pgvector \
           -p 5432:5432 \
           --replace \
           pgvector/pgvector:pg17
```

On a different shell:

```shell
podman exec -it pgvector /bin/bash
```

Inside the container:

```shell
psql -U app -c "CREATE EXTENSION VECTOR;"
```

### Enable the Feast repository

Go to `/rec-sys-app/feature_repo`, and then:

```shell
feast apply
```

The first time you run it, you should see:

```shell
No project found in the repository. Using project name feast_edb_rec_sys defined in feature_store.yaml
Applying changes for project feast_edb_rec_sys
Deploying infrastructure for item_embedding
Deploying infrastructure for user_features
Deploying infrastructure for item_textual_features_embed
Deploying infrastructure for user_embedding
Deploying infrastructure for interactions_features
Deploying infrastructure for item_features
Deploying infrastructure for user_items
```

### Run the driver

Go to `/rec-sys-app`, and then:

```shell
python driver.py
```
