import torch.nn as nn

def make_normalization(norm_type):
    if norm_type == 'batch':
        return nn.BatchNorm1d
    elif norm_type == 'layer':
        return nn.LayerNorm
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")    

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, normalization, hidden_factor, hidden_dropout, residual_dropout):
        super(ResNetBlock, self).__init__()
        hidden_dim = int(input_dim * hidden_factor)
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = normalization(hidden_dim)
        self.dropout1 = nn.Dropout(hidden_dropout)
        
        self.layer2 = nn.Linear(hidden_dim, input_dim)
        self.norm2 = normalization(input_dim)
        self.dropout2 = nn.Dropout(residual_dropout)
        
        self.activation = nn.ReLU()

        self.ff = nn.Sequential(
            normalization(input_dim),
            nn.Linear(input_dim, input_dim),
            self.activation,
            nn.Dropout(residual_dropout),
            nn.Linear(input_dim, input_dim),
            nn.Dropout(residual_dropout)
        )
    def forward(self, x):
        return x + self.ff(x)
    

class ResNetRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks, hidden_factor=4, hidden_dropout=0.1, residual_dropout=0.1, normalization='layer'):
        super(ResNetRegressor, self).__init__()
        norm_layer = make_normalization(normalization)
        
        self.initial_layer = nn.Linear(input_dim, input_dim)
        
        self.blocks = nn.ModuleList([
            ResNetBlock(input_dim, norm_layer, hidden_factor, hidden_dropout, residual_dropout)
            for _ in range(num_blocks)
        ])
        
        self.final_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        return x