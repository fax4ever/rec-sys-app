import torch
import torch.nn as nn
from torch import Tensor
from models.item_tower import ItemTower
from models.user_tower import UserTower

class TwoTowerModel(nn.Module):
    def __init__(self, item_tower: ItemTower, user_tower: UserTower):
        super().__init__()
        self.item_tower = item_tower
        self.user_tower = user_tower
        
        
        
    def forward(self, proccesed_items_dict: Tensor, proccesed_users_dict: Tensor, proccesed_real_intercations: Tensor):
        # TODO implement me if needed
        pass

