import torch
from torch import nn

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
    
    def forward(self, x):
        return x.unsqueeze(-1)*self.weight + self.bias

class FTTransformer(nn.Module):
    def __init__(self, 
                 n_features,
                 output_dim = 1,
                 d_token = 32,
                 n_heads = 4,
                 n_layers = 4, 
                 dropout = 0.1,
                 ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dropout=dropout,
            batch_first = True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 num_layers=n_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, output_dim)
        )
    
    def forward(self, x):
        B = x.size(0)
        x = self.tokenizer(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)

        cls_out = x[:, 0]

        return self.head(cls_out)
