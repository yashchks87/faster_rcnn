import numpy as np
import numpy as xp

def generate_anchor_base(base_size = 16, ratios = [0.5, 1, 2], anchor_scales = [8, 16, 32]):
    """
        Explanation:
            This function generates references anchor boxes over each and every pixel on the feature map.
            Please read this page as well for better understanding
            https://www.perplexity.ai/search/what-is-generate-anchor-base-f-C.hiWrC_Td.e_A3At.CAOQ
            RPN uses this boxes as part of refrence boxes.
        Args:
            base_size (int): The width and height of the reference window. 
            TODO: Check what it exactly means.
    """
    py = base_size / 2.
    px = base_size / 2.

    # Generate reference array for future hold.
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype = np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            # TODO: Why do we do divide by 1?
            w = base_size * anchor_scales[j] * np.sqrt(1 / ratios[i])
            # Just generates index to iterate over all the anchor boxes from anchor_base.
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    return anchor_base



