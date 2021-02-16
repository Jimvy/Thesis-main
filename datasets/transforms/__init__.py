from torch import Tensor


class Lighting:
    r"""
    Lighting noise, for AlexNet - style PCA-based noise.

    Comes from antspy/quantized_distillation/blob/master/datasets/torchvision_extension.py,
    which itself comes from pytorch/vision/pull/27.
    """
    def __init__(self, alphastd, eigval: Tensor, eigvec: Tensor):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img: Tensor):
        if self.alphastd == 0:
            return img
        alpha = img.new_tensor(img).resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))
