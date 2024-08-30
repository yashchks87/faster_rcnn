import torch
import torch.nn as nn
import torch.nn.functional as F
from bbox_tools import generate_anchor_base
from creator_tool import ProposalCreator
import numpy as np

class RegionProposalNetwork(nn.Module):
    # Region Proposal Network proposes are the regions which are likely to contain objects
    def __init__(self, 
                 in_channels = 512, mid_channels = 512, ratios = [0.5, 1, 2],
                 anchor_scales = [8, 16, 32], feat_stride = 16,
                 proposal_creator_params = dict()):
        super(RegionProposalNetwork, self).__init__()
        # This generates anchor boxes as per given 
        # We need to check still that how that being produced here because can we do that generates from clustering or not?
        # Here it looks like it is being generated from the ratios and scales.
        self.anchor_base = generate_anchor_base(
            anchor_scales = anchor_scales,
            ratios = ratios
        )
        self.feat_stride = feat_stride
        # TODO: Check why we are using this?
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        # TODO: Activate it once we have anchor code ready
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, 2 * n_anchor, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, 4 * n_anchor, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
    
    def forward(self, x, img_size, scale = 1.):
        n, _, hh, ww = x.shape
        # n -> Number of images(batch size), _ -> Number of channels, hh -> Height of the feature map, ww -> Width of the feature map
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), 
            self.feat_stride, 
            hh, 
            ww)
        # Review this code block again
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim = 4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        # Start extracting rois
        rois = list()
        roi_indices = list()
        # We will itereate over all the images in the batch
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size, scale = scale)
            batch_index = i * np.ones((len(roi),), dtype = np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        # Simple conversion from list to numpy array
        rois = np.concatenate(rois, axis = 0)
        roi_indices = np.concatenate(roi_indices, axis = 0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


    
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Please read this function carefully and this is explanation in this link:
    # https://www.perplexity.ai/page/what-is-enumerate-shifted-anch-JhT6ugbPSdeNPyfeSNka5Q
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    # Check ravel info here: https://www.perplexity.ai/search/difference-between-ravel-vs-fl-ESuO2LboQM28.QN.fO36mA
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis = 1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

def normal_init(m, mean, stddev, truncated = False):
    # This function is used to initialize weights and bias of the convolutional layers
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()