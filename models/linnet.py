import torch
import torch.nn as nn

from position_embedding import SinusoidalEmbedding
from layers import LinearBlock


class LinNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128
    ) -> None:
        super(LinNet, self).__init__()

        self.time_mlp = SinusoidalEmbedding(emb_size, scale=1.0)
        self.input_mlp1 = SinusoidalEmbedding(emb_size, scale=25.0)  # https://bmild.github.io/fourfeat/
        self.input_mlp2 = SinusoidalEmbedding(emb_size, scale=25.0)  # https://bmild.github.io/fourfeat/

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(LinearBlock(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x
