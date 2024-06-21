import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


class LossService:
    @staticmethod
    def iou_per_class(y_true, y_pred, smooth=1e-6):
        if y_true.dim() != 4 or y_pred.dim() != 4:
            raise ValueError("y_true and y_pred must be B x C x H x W tensors")

        intersection = torch.sum(y_true * y_pred, dim=(0, 2, 3))
        union = torch.sum(y_true, dim=(0, 2, 3)) + torch.sum(y_pred, dim=(0, 2, 3)) - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou

    @staticmethod
    def focal_loss(y_true, y_pred, gamma=2., alpha=4.):
        epsilon = 1.e-9
        model_out = y_pred + epsilon
        ce = -y_true * torch.log(model_out)
        weight = y_true * ((1 - model_out) ** gamma)
        fl = alpha * weight * ce
        reduced_fl = torch.max(fl, dim=-1)[0]  # Using max along the classes dimension
        return torch.mean(reduced_fl)

    @staticmethod
    def ssim_loss(y_true, y_pred):
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        ssim(y_pred, y_true)

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1.e-9):
        intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
        union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3])
        return torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
