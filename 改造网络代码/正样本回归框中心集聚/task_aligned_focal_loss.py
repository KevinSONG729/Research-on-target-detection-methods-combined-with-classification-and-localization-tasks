import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss
import logging
import numpy

numpy.set_printoptions(threshold=70000)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename='/media/store1/sxw/code/tood/test.log',
                    filemode='w')

# python version no_sigmoid
def focal_loss_with_prob(prob,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    target_one_hot = prob.new_zeros(len(prob), len(prob[0]) + 1).scatter_(1, target.unsqueeze(1), 1)[:, :-1]

    flatten_alpha = torch.empty_like(prob).fill_(1 - alpha)
    flatten_alpha[target_one_hot == 1] = alpha

    pt = torch.where(target_one_hot == 1, prob, 1 - prob)

    ce_loss = F.binary_cross_entropy(
        prob, target_one_hot, reduction='none')
    loss = flatten_alpha * torch.pow(1 - pt, gamma) * ce_loss

    if weight is not None:
        loss = weight_reduce_loss(loss, weight.reshape(-1, 1), reduction, avg_factor)
    else:
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

# python version for task aligned focal loss
def task_aigned_focal_loss(prob,
                       target,
                       alignment_metric,
                       weight=None,
                       gamma=2.0,
                       reduction='mean',
                       avg_factor=None):
    # logging.debug('target is' + str(target))
    # logging.debug('prob is' + str(prob.tolist()))
    target_one_hot = prob.new_zeros(len(prob), len(prob[0]) + 1).scatter_(1, target.unsqueeze(1), 1)[:, :-1]
    # logging.debug('target_one_hot nonzero num is' + str(torch.count_nonzero(target_one_hot).item()))
    soft_label = alignment_metric.unsqueeze(-1) * target_one_hot
    # print(soft_label.size())
    ce_loss = F.binary_cross_entropy(
        prob, soft_label, reduction='none')
    # print(ce_loss.size())
    loss = torch.pow(torch.abs(soft_label - prob), gamma) * ce_loss
    # print(loss.sum(1).size(), weight.size())
    loss = weight_reduce_loss(loss.sum(1), weight, reduction, avg_factor)
    # logging.debug('focal_loss prob require_grad is' + str(prob.requires_grad))
    # logging.debug('focal_loss require_grad is' + str(loss.requires_grad))
    return loss


@LOSSES.register_module()
class FocalLossWithProb(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLossWithProb, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                prob,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * focal_loss_with_prob(
                prob,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls



@LOSSES.register_module()
class TaskAlignedFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(TaskAlignedFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                prob,
                target,
                alignment_metric,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * task_aigned_focal_loss(
                prob,
                target,
                alignment_metric,
                weight,
                gamma=self.gamma,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
