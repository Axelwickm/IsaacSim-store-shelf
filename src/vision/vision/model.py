import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


NUM_QUERIES = 64
LATENT_DIM = 4
PATCH_SIZE = 32
HIDDEN_DIM = 96


class QueryDepthIdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.backbone = backbone.features
        self.feature_proj = nn.Conv2d(576, HIDDEN_DIM, kernel_size=1)
        self.query_embed = nn.Parameter(torch.randn(NUM_QUERIES, HIDDEN_DIM) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN_DIM,
            nhead=4,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.slot_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN_DIM, 10 + LATENT_DIM),
        )

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        features = self.feature_proj(features)
        batch_size = features.shape[0]
        tokens = features.flatten(2).transpose(1, 2)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, tokens)
        slot_params = self.slot_head(decoded)

        presence_logits = slot_params[..., 0:1]
        centers = torch.tanh(slot_params[..., 1:3])
        depth = torch.sigmoid(slot_params[..., 3:4]) * 5.0
        sizes = torch.sigmoid(slot_params[..., 4:6]) * 0.08 + 0.015
        auxiliary = slot_params[..., 6:8]
        value_logits = slot_params[..., 8:10]
        latent = slot_params[..., 10:]
        return {
            "presence_logits": presence_logits,
            "presence_probs": torch.sigmoid(presence_logits),
            "centers": centers,
            "depth": depth,
            "sizes": sizes,
            "auxiliary": auxiliary,
            "latent": latent,
            "value_logits": value_logits,
            "value_probs": torch.sigmoid(value_logits),
        }


def create_query_model():
    return QueryDepthIdentityModel()


def model_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
