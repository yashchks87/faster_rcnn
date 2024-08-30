import torch.nn as nn

class FasterRCNN(nn.Module):
    # We need this base class to be called all the time to make sure it can take and accept
    # values.
    def __init__(self,
                 extractor,
                 rpn, head,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')
        # Define your model architecture here
    
    def use_preset(self, preset):
        # pass
        if preset == 'visualize':
            # If we are using visualize aspect only then we set the following parameters
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        # elif:
        #     pass


    def forward(self, x):
        # Simple forward pass which will take data forward
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices
    
    def train(self, mode=True):
        pass
        # Define the training logic of your model here
        # This is where you define how the model is trained
    
    def eval(self):
        pass
        # Define the evaluation logic of your model here
        # This is where you define how the model is evaluated