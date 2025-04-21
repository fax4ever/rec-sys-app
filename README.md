# Recommendation System Application
## Overview
This application implements a recommendation system using a two-tower neural network architecture, integrated with Feast for feature management. It generates synthetic datasets for users, items, and interactions, trains a model to produce user and item embeddings, and provides personalized item recommendations. 

## Features

* Data Generation: Creates synthetic datasets for users, items, positive interactions, and negative interactions using dataset_gen.py.
* Feature Store: Utilizes Feast to manage and serve features, with configurations defined in feature_repo/.
* Model Training: Trains a two-tower model (UserTower and ItemTower) to generate embeddings, implemented in models/.
* Filtering: Applies rule-based filtering (availability, demographic, history, and contextual) to refine recommendations (models/filtering.py).