from models.two_tower import TwoTowerModel
import pandas as pd
from models.data_util import preproccess_pipeline, UserItemMagnitudeDataset
from models.user_tower import UserTower
from models.item_tower import ItemTower
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def train_two_tower(item_tower: ItemTower, user_tower: UserTower, item_df: pd.DataFrame, user_df: pd.DataFrame, interaction_df_pos: pd.DataFrame, interaction_df_neg: pd.DataFrame, return_epoch_losses: bool=False, n_epochs: int = 10):

    dataset = preproccess_pipeline(item_df, user_df, interaction_df_pos, interaction_df_neg)
    two_tower_model = TwoTowerModel(item_tower, user_tower)
    
    epoch_losses = _train(dataset, two_tower_model, n_epochs=n_epochs)
    if return_epoch_losses:
        return epoch_losses
    
def _train(dataset: UserItemMagnitudeDataset, two_tower_model: TwoTowerModel, n_epochs: int = 10, 
           device: str = 'cpu', batch_size: int = 256):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(two_tower_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Assuming magnitude prediction is a regression task
    
    # Set model to training mode
    two_tower_model.to(device)
    two_tower_model.train()
    
    # Store losses for each epoch
    epoch_losses = []
    
    # Training loop over epochs
    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for items, users, magnitude in dataloader:
            # Move data to specified device
            items = {key: value.to(device) for key, value in items.items()}
            users = {key: value.to(device) for key, value in users.items()}
            magnitude = magnitude.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = two_tower_model(items, users)
            
            # Calculate loss
            loss = criterion(predictions, magnitude)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate and store average loss for the epoch
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        
        # Optional: Print progress
        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    return epoch_losses
