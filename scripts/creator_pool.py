import torch
import torch.nn as nn

class ProposalCreator:
    """
        Explanation:
            This class is responsible for creating proposal and also does NMS and make sure 
            after NMS how many proposal we creates.
        Args:
            parent_model: The parent model of the proposal creator.
            nms_thresh: The threshold value for non maximum suppression.
    """
    def __init__(
                    self, 
                    parent_model,
                    nms_thresh = 0.7,
                    n_train_pre_nms = 12000,
                    n_train_post_nms = 2000,
                    n_test_pre_nms = 6000,
                    n_test_post_nms = 300,
                    min_size = 16
                ):
        self.parent_model = parent_model
        self.nms_thres = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
        
    # Call method is for calling the instance of the class like a function.
    def __call__(self, loc, score, anchor, img_size, scale = 1.):
        """
            This function proposes ROIS from given features.
            
        """
        pass