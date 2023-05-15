import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss
import logging
import math

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename='test.log',
                    filemode='w')

# python version for task aligned focal loss version_2
def task_aigned_focal_loss_v2(prob,
                       target,
                       alignment_metric,
                       stride,
                       batch_size,
                       img_size,
                       num_classes,
                       weight=None,
                       gamma=2.0,
                       reduction='mean',
                       avg_factor=None):
    # logging.debug('prob is ' + str(prob.size()))
    logging.debug('target is ' + str(target.size()))
    # logging.debug('target_value is ' + str(target))
    # logging.debug('stride is ' + str(stride))
    # logging.debug('batch_size is ' + str(batch_size))
    # logging.debug('img_size is' + str(img_size.size()))
    # logging.debug('img_size[0] is' + str(img_size[0][0]) + ' ' + str(img_size[0][1]))
    # logging.debug('weight is ' + str(weight.size()))
    # logging.debug('weight_value is ' + str(weight))
    target_one_hot = prob.new_zeros(len(prob), len(prob[0]) + 1).scatter_(1, target.unsqueeze(1), 1)[:, :-1]
    # logging.debug('target_one_hot is ' + str(target_one_hot.size()))
    # logging.debug('target_one_hot[0] is ' + str(target_one_hot[0]))
    soft_label = alignment_metric.unsqueeze(-1) * target_one_hot
    # logging.debug('soft_label is ' + str(soft_label.size()))
    # print(soft_label.size())
    ce_loss = F.binary_cross_entropy(
        prob, soft_label, reduction='none')
    # print(ce_loss.size())
    # logging.debug('ce_loss is ' + str(ce_loss.size()))
    loss = torch.pow(torch.abs(soft_label - prob), gamma) * ce_loss
    # print(loss.sum(1).size(), weight.size())
    # loss = weight_reduce_loss(loss.sum(1), weight, reduction, avg_factor)
    # logging.debug('loss is ' + str(loss.size()))
    loss = weight_reduce_loss(loss.sum(1), weight, reduction, avg_factor)
    distance_loss = 0
    pos_idx_list = batch_img_pos_idx(target, stride, batch_size, img_size, num_classes)
    for single_img_pos in pos_idx_list:
        for pos_idx in single_img_pos:
            distance_loss += euclidean_dist(pos_idx, pos_idx)
    logging.debug('distance_loss is ' + str(distance_loss))
    return loss


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n] / int
    """
 
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    dist.addmm_(1, -2, x, y.t())
    #dist.addmm_(x, y.t(), 1, -2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return torch.sum(dist)/2


def batch_img_pos_idx(target, stride, batch_size, img_size_list, num_classes):
    """
    Args:
        target (torch.Tensor): The learning label of the prediction.
        stride (int): downsamping rate of the img.
        batch_size (int): the number of the batch size.
        img_size_list (List(Tensor)): the img_meta size of shape (b, H, W).
        num_classes (int): the number of the classes in dataset.
    Returns:
        pos_idx_list (List(List(Tensor))): the coordinate of the pos samples in featuremap.
            the outer List is the different imgs, and the inner List is the different gts. 
    """
    img_size_list = [img_size.float() for img_size in img_size_list]
    for img_size in img_size_list:
        img_size[0] = math.ceil(img_size[0]/stride)
        img_size[1] = math.ceil(img_size[1]/stride)
    target = target.detach()
    target_tuple = target.split([img_size[0].int()*img_size[1].int() for img_size in img_size_list], dim=0)
    pos_idx_list = []
    for idx, target in enumerate(target_tuple):
        target = target.reshape((img_size_list[idx][0].int(), img_size_list[idx][1].int()))
        res = {}
        for i in range(target.size(0)):
	        for j in range(target.size(-1)):
		        if(target[i][j].item()!=num_classes):
			        if target[i][j].item() not in res:
				        res[target[i][j].item()] = torch.tensor([[i, j]])
			        else:
				        res[target[i][j].item()] = torch.cat([res[target[i][j].item()], torch.tensor([[i,j]])], dim=0)
        pos_idx_list.append([v for v in res.values()])
    return pos_idx_list


@LOSSES.register_module()
class TaskAlignedFocalLossv2(nn.Module):

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
        super(TaskAlignedFocalLossv2, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                prob,
                target,
                alignment_metric,
                stride,
                batch_size,
                img_size,
                num_classes,
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
        logging.debug('reduction is ' + str(reduction))
        if self.use_sigmoid:
            loss_cls = self.loss_weight * task_aigned_focal_loss_v2(
                prob,
                target,
                alignment_metric,
                stride,
                batch_size,
                img_size,
                num_classes,
                weight,
                gamma=self.gamma,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls