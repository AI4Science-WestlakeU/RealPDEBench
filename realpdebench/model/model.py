import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise Exception(NotImplementedError)

    def train_loss(self, input, target):
        raise Exception(NotImplementedError)

    def load_checkpoint(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

        meta_data = {
            'all_train_losses': checkpoint['train_losses'],
            'all_val_losses': checkpoint['val_losses'],
            'iteration': checkpoint['iteration'],
            'best_iteration': checkpoint['best_iteration'],
            'best_val_loss': checkpoint['best_val_loss']
        }

        return meta_data
