import torch
import torch.nn as nn
from bbox_tools import generate_anchor_base
from creator_tool import ProposalCreator

class RegionProposalNetwork(nn.Module):
    # Region Proposal Network proposes are the regions which are likely to contain objects
    def __init__(
            self, in_channels = 512, mid_channels = 512, feat_stride = 16,
            anchor_scales = [8, 16, 32], ratios = [0.5, 1, 2],
            proposal_creator_params = dict()
            ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales = anchor_scales, ratios = ratios
        )
        self.feat_stride = feat_stride
        # self.proposal_layer = ProposalCreator()
        # 3 convolutional layers
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size = 3, stride = 1, padding = 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

