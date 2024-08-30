import torch
import torch.nn as nn
import torchvision
from torchvision.ops import RoIPool
from region_proposal_network import RegionProposalNetwork

def decom_vgg16():
    # Import vgg16 model
    # TODO: Change this to weather we want to use pre trained weights or not 
    model = torchvision.models.vgg16()
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)

    # Remove last layer -> Why? find out.
    del classifier[6]
    classifier = nn.Sequential(*classifier)

    # Freeze frst 4 convolutional layers
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    
    return nn.Sequential(*features), classifier

class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self,
                 n_fg_class = 20,
                 ratios = [0.5, 1, 2],
                 anchor_scales = [8, 16, 32]):
        
        # extractor: Simple feature extractors part of convolutional layers
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios = ratios, 
            anchor_scales = anchor_scales,
            feat_stride = self.feat_stride
        )

        head = VGG16ROIHead(
            n_class = n_fg_class + 1,
            roi_size=7,
            spatial_scale =  (1. / self.feat_stride),
            classifier = classifier
        )




class VGG16ROIHead(nn.Module):
    # This will ingest the output of ROI generation layer which generates the region proposals
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16ROIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        # Simple classification layar which generates score.
        self.score = nn.Linear(4096, n_class)

        # Normal initialization for regression layer and classification layer
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """
            Args:
                x: (N, C, H, W) feature map -> Output of VGG16
                rois: (R, 4) [x1, y1, x2, y2] -> Region proposals
                roi_indices: (R, 1) index of image to which roi belongs
        """
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_score = self.score(fc7)
        return roi_cls_locs, roi_score

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()