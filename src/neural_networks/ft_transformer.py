import torch
from torch import nn

class SharedAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
    
    def forward(self, x):
        return x.unsqueeze(-1)*self.weight + self.bias

class FFN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

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

        self.attn =  SharedAttention(d_token, n_heads)

        self.ffns = nn.ModuleList([FFN(d_token, dropout) for _ in range(n_layers)])

        self.norm_attn = nn.LayerNorm(d_token)

        self.norm_ffns = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, output_dim)
        )
    
    def forward(self, x):
        B = x.size(0)
        x = self.tokenizer(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for ff, norm in zip(self.ffns, self.norm_ffns):
            x = x + self.attn(self.norm_attn(x))
            x = x + ff(norm(x))

        cls_out = x[:, 0]

        return self.head(cls_out)
