from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.models.optical_flow.raft import ResidualBlock, CorrBlock, RecurrentBlock, UpdateBlock, BottleneckBlock
from torchvision.models.optical_flow.raft import FeatureEncoder, MotionEncoder, FlowHead, MaskPredictor
from torchvision.models.optical_flow.raft import Raft_Large_Weights, RAFT, Raft_Small_Weights
from torchvision.models.optical_flow._utils import make_coords_grid, upsample_flow
import torch
import torch.nn.functional as F


class RAFT_mo(RAFT):
    def forward_wo_upsample(self, image1, image2, num_flow_updates: int = 12):

        batch_size, _, h, w = image1.shape
        if (h, w) != image2.shape[-2:]:
            raise ValueError(f"input images should have the same shape, instead got ({h}, {w}) != {image2.shape[-2:]}")
        if not (h % 8 == 0) and (w % 8 == 0):
            raise ValueError(f"input image H and W should be divisible by 8, instead got {h} (h) and {w} (w)")

        fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
        if fmap1.shape[-2:] != (h // 8, w // 8):
            raise ValueError("The feature encoder should downsample H and W by 8")

        self.corr_block.build_pyramid(fmap1, fmap2)

        context_out = self.context_encoder(image1)
        if context_out.shape[-2:] != (h // 8, w // 8):
            raise ValueError("The context encoder should downsample H and W by 8")

        # As in the original paper, the actual output of the context encoder is split in 2 parts:
        # - one part is used to initialize the hidden state of the recurent units of the update block
        # - the rest is the "actual" context.
        hidden_state_size = self.update_block.hidden_state_size
        out_channels_context = context_out.shape[1] - hidden_state_size
        if out_channels_context <= 0:
            raise ValueError(
                f"The context encoder outputs {context_out.shape[1]} channels, but it should have at strictly more than hidden_state={hidden_state_size} channels"
            )
        hidden_state, context = torch.split(context_out, [hidden_state_size, out_channels_context], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = F.relu(context)

        coords0 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
        coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)

        flow_predictions = []
        flow_downsamples = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
            corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)

            flow = coords1 - coords0
            hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)

            coords1 = coords1 + delta_flow
            gene_flow = coords1 - coords0
            flow_downsamples.append(gene_flow)

            # up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state)
            # upsampled_flow = upsample_flow(flow=(gene_flow), up_mask=up_mask)
            # flow_predictions.append(upsampled_flow)

        return flow_downsamples #, flow_predictions


def _raft_moified(
    *,
    weights=None,
    progress=False,
    # Feature encoder
    feature_encoder_layers,
    feature_encoder_block,
    feature_encoder_norm_layer,
    # Context encoder
    context_encoder_layers,
    context_encoder_block,
    context_encoder_norm_layer,
    # Correlation block
    corr_block_num_levels,
    corr_block_radius,
    # Motion encoder
    motion_encoder_corr_layers,
    motion_encoder_flow_layers,
    motion_encoder_out_channels,
    # Recurrent block
    recurrent_block_hidden_state_size,
    recurrent_block_kernel_size,
    recurrent_block_padding,
    # Flow Head
    flow_head_hidden_size,
    # Mask predictor
    use_mask_predictor,
    **kwargs,
):
    feature_encoder = kwargs.pop("feature_encoder", None) or FeatureEncoder(
        block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer
    )
    context_encoder = kwargs.pop("context_encoder", None) or FeatureEncoder(
        block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer
    )

    corr_block = kwargs.pop("corr_block", None) or CorrBlock(num_levels=corr_block_num_levels, radius=corr_block_radius)

    update_block = kwargs.pop("update_block", None)
    if update_block is None:
        motion_encoder = MotionEncoder(
            in_channels_corr=corr_block.out_channels,
            corr_layers=motion_encoder_corr_layers,
            flow_layers=motion_encoder_flow_layers,
            out_channels=motion_encoder_out_channels,
        )

        # See comments in forward pass of RAFT class about why we split the output of the context encoder
        out_channels_context = context_encoder_layers[-1] - recurrent_block_hidden_state_size
        recurrent_block = RecurrentBlock(
            input_size=motion_encoder.out_channels + out_channels_context,
            hidden_size=recurrent_block_hidden_state_size,
            kernel_size=recurrent_block_kernel_size,
            padding=recurrent_block_padding,
        )

        flow_head = FlowHead(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size)

        update_block = UpdateBlock(motion_encoder=motion_encoder, recurrent_block=recurrent_block, flow_head=flow_head)

    mask_predictor = kwargs.pop("mask_predictor", None)
    if mask_predictor is None and use_mask_predictor:
        mask_predictor = MaskPredictor(
            in_channels=recurrent_block_hidden_state_size,
            hidden_size=256,
            multiplier=0.25,  # See comment in MaskPredictor about this
        )

    model = RAFT_mo(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_block=corr_block,
        update_block=update_block,
        mask_predictor=mask_predictor,
        **kwargs,  # not really needed, all params should be consumed by now
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def raft_large_modified(*, weights: None, progress=True, **kwargs) -> RAFT:
    """RAFT model 
    modified from torchvision.models.optical_flow.raft.raft_large
    """
    weights = Raft_Large_Weights.verify(weights)

    return _raft_moified(
        weights=weights,
        progress=progress,
        # Feature encoder
        feature_encoder_layers=(64, 64, 96, 128, 256),
        feature_encoder_block=ResidualBlock,
        feature_encoder_norm_layer=InstanceNorm2d,
        # Context encoder
        context_encoder_layers=(64, 64, 96, 128, 256),
        context_encoder_block=ResidualBlock,
        context_encoder_norm_layer=BatchNorm2d,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(256, 192),
        motion_encoder_flow_layers=(128, 64),
        motion_encoder_out_channels=128,
        # Recurrent block
        recurrent_block_hidden_state_size=128,
        recurrent_block_kernel_size=((1, 5), (5, 1)),
        recurrent_block_padding=((0, 2), (2, 0)),
        # Flow head
        flow_head_hidden_size=256,
        # Mask predictor
        use_mask_predictor=True,
        **kwargs,
    )
    

def raft_small_modified(*, weights: None, progress=True, **kwargs) -> RAFT:
    """RAFT model 
    modified from torchvision.models.optical_flow.raft.raft_small
    """
    weights = Raft_Small_Weights.verify(weights)

    return _raft_moified(
        weights=weights,
        progress=progress,
        # Feature encoder
        feature_encoder_layers=(32, 32, 64, 96, 128),
        feature_encoder_block=BottleneckBlock,
        feature_encoder_norm_layer=InstanceNorm2d,
        # Context encoder
        context_encoder_layers=(32, 32, 64, 96, 160),
        context_encoder_block=BottleneckBlock,
        context_encoder_norm_layer=None,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=3,
        # Motion encoder
        motion_encoder_corr_layers=(96,),
        motion_encoder_flow_layers=(64, 32),
        motion_encoder_out_channels=82,
        # Recurrent block
        recurrent_block_hidden_state_size=96,
        recurrent_block_kernel_size=(3,),
        recurrent_block_padding=(1,),
        # Flow head
        flow_head_hidden_size=128,
        # Mask predictor
        use_mask_predictor=False,
        **kwargs,
    )