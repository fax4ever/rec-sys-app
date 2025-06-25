# Recommendation System Application
## Overview
This application implements a recommendation system using a two-tower neural network architecture, integrated with Feast for feature management. It generates synthetic datasets for users, items, and interactions, trains a model to produce user and item embeddings, and provides personalized item recommendations. 

## Features

* Data Generation: Creates synthetic datasets and images under `generation` directory
* Feature Store: Utilizes Feast to manage and serve features, with configurations defined in feature_repo/.
* Model Training: Trains a two-tower model (UserTower and ItemTower) to generate embeddings, implemented in models/.
* Filtering: Applies rule-based filtering (availability, demographic, history, and contextual) to refine recommendations (models/filtering.py).

## Data generation

The application includes two main data generation components:

### Dataset Generation (`dataset_gen_amazon.py`)

Generates synthetic e-commerce data including:

* Users: Creates user profiles with IDs, signup dates, and category preferences
* Items: Generates product data with categories, prices, ratings, and descriptions based on Amazon-like structure
* Interactions: Simulates user-item interactions (views, purchases, ratings) with preference-based bias

The generated data is saved in parquet format under `feature_repo/data/`:
* `recommendation_users.parquet`
* `recommendation_items.parquet`
* `recommendation_interactions.parquet`

```bash
# Generate synthetic data
python generation/dataset_gen_amazon.py --n_users 1000 --n_items 5000 --n_interactions 20000
```

### Image Generation (`generate_images.py`)

Generates product images using Stable Diffusion:

* Uses the RunwayML Stable Diffusion v1.5 model
* Creates images based on product descriptions
* Supports both CPU and CUDA-enabled GPU processing
* Saves generated images in PNG format under `generation/data/generated_images/`

To generate the datasets, run:

```bash
# Generate product images (requires PyTorch and diffusers)
python generation/generate_images.py
```