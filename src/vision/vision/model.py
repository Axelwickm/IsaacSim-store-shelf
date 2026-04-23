import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


NUM_QUERIES = 64
LATENT_DIM = 4
PATCH_SIZE = 32
HIDDEN_DIM = 96
IDENTITY_ALPHA_EPSILON = 1e-3


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
            nn.Linear(HIDDEN_DIM, 8 + LATENT_DIM),
        )
        self.patch_decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, PATCH_SIZE * PATCH_SIZE),
        )

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        features = self.feature_proj(features)
        batch_size, _, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, tokens)
        slot_params = self.slot_head(decoded)

        presence_logits = slot_params[..., 0]
        centers = torch.tanh(slot_params[..., 1:3])
        depth = torch.sigmoid(slot_params[..., 3:4]) * 5.0
        sizes = torch.sigmoid(slot_params[..., 4:6]) * 0.08 + 0.015
        alpha_scale = torch.sigmoid(slot_params[..., 6:7])
        depth_scale = torch.sigmoid(slot_params[..., 7:8])
        latent = slot_params[..., 8:]
        patch_alpha_logits = self.patch_decoder(latent).view(
            batch_size, NUM_QUERIES, PATCH_SIZE, PATCH_SIZE
        )
        presence = torch.sigmoid(presence_logits).unsqueeze(-1).unsqueeze(-1)
        patch_alpha = (
            torch.sigmoid(patch_alpha_logits)
            * alpha_scale.squeeze(-1).unsqueeze(-1).unsqueeze(-1)
            * presence
        )

        rendered_identity, rendered_depth, rendered_occupancy = render_billboards(
            patch_alpha=patch_alpha,
            centers=centers,
            depth=depth,
            sizes=sizes,
            depth_scale=depth_scale,
            image_height=pixel_values.shape[-2],
            image_width=pixel_values.shape[-1],
        )
        return {
            "presence_logits": presence_logits,
            "centers": centers,
            "depth": depth,
            "sizes": sizes,
            "patch_alpha_logits": patch_alpha_logits,
            "patch_alpha": patch_alpha,
            "predicted_identity": rendered_identity,
            "predicted_depth": rendered_depth,
            "predicted_occupancy": rendered_occupancy,
        }


def render_billboards(
    patch_alpha,
    centers,
    depth,
    sizes,
    depth_scale,
    image_height: int,
    image_width: int,
):
    device = patch_alpha.device
    batch_size, num_queries, patch_h, patch_w = patch_alpha.shape
    ys = torch.linspace(-1.0, 1.0, image_height, device=device)
    xs = torch.linspace(-1.0, 1.0, image_width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.view(1, 1, image_height, image_width)
    grid_y = grid_y.view(1, 1, image_height, image_width)

    centers_x = centers[..., 0].view(batch_size, num_queries, 1, 1)
    centers_y = centers[..., 1].view(batch_size, num_queries, 1, 1)
    sizes_x = sizes[..., 0].view(batch_size, num_queries, 1, 1)
    sizes_y = sizes[..., 1].view(batch_size, num_queries, 1, 1)

    local_x = ((grid_x - centers_x) / sizes_x).clamp(-1.1, 1.1)
    local_y = ((grid_y - centers_y) / sizes_y).clamp(-1.1, 1.1)
    sample_grid = torch.stack([local_x, local_y], dim=-1).view(
        batch_size * num_queries, image_height, image_width, 2
    )
    sampled_alpha = torch.nn.functional.grid_sample(
        patch_alpha.view(batch_size * num_queries, 1, patch_h, patch_w),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch_size, num_queries, image_height, image_width)

    sorted_indices = torch.argsort(depth.squeeze(-1), dim=1, descending=True)
    sorted_alpha = torch.gather(
        sampled_alpha,
        1,
        sorted_indices[:, :, None, None].expand(-1, -1, image_height, image_width),
    )
    sorted_depth = torch.gather(depth, 1, sorted_indices[:, :, None])
    sorted_depth = sorted_depth.unsqueeze(-1).expand(-1, -1, image_height, image_width)
    sorted_identity = (
        sorted_indices.to(sampled_alpha.dtype)[:, :, None, None].expand(
            -1, -1, image_height, image_width
        )
        + 1.0
    )

    rendered_identity = torch.zeros(
        batch_size, image_height, image_width, device=device, dtype=sampled_alpha.dtype
    )
    rendered_depth = torch.zeros(
        batch_size, image_height, image_width, device=device, dtype=sampled_alpha.dtype
    )
    transmittance = torch.ones(
        batch_size, image_height, image_width, device=device, dtype=sampled_alpha.dtype
    )
    assigned = torch.zeros(
        batch_size, image_height, image_width, device=device, dtype=torch.bool
    )

    for query_index in range(num_queries):
        alpha = sorted_alpha[:, query_index]
        contribution = alpha * transmittance
        rendered_depth = rendered_depth + contribution * sorted_depth[:, query_index]
        newly_assigned = (contribution > IDENTITY_ALPHA_EPSILON) & (~assigned)
        rendered_identity = torch.where(
            newly_assigned,
            sorted_identity[:, query_index],
            rendered_identity,
        )
        assigned = assigned | newly_assigned
        transmittance = transmittance * (1.0 - alpha)

    rendered_occupancy = (1.0 - transmittance).clamp(0.0, 1.0)
    visible_mask = rendered_occupancy > 0.05
    rendered_identity = rendered_identity * visible_mask
    rendered_depth = rendered_depth * visible_mask
    rendered_depth = rendered_depth.unsqueeze(1) * depth_scale.mean(
        dim=1, keepdim=True
    ).unsqueeze(-1)
    return rendered_identity, rendered_depth, rendered_occupancy.unsqueeze(1)


def create_query_model():
    return QueryDepthIdentityModel()


def model_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
