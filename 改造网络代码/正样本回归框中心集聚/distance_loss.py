from dis import dis
from statistics import mean
import mmcv
import torch
import torch.nn as nn
from ..builder import LOSSES
import math
import logging

device = torch.device('cuda:0')

@mmcv.jit(derivate=True, coderize=True)
def pos_idx_groupbyClass(pos_target):
    """
    Args:
        pos_target(torch.Tensor): the classes of the positive sampling points in one level(one batch).
                which has shape[num_pos,]
    Returns:
        grouped_Inds_List(List[Tensor]): the element in List is the inds of one class.
                the length of the List is num_class, the size of the Tensor is num_point_in_one_class.
    """
    grouped_inds_list = []
    target_set = torch.unique(pos_target)
    for c in target_set:
        grouped_inds_list.append(torch.nonzero(pos_target==c))
    return grouped_inds_list

@mmcv.jit(derivate=True, coderize=True)
def euclidean_dist(x):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
    Returns:
        dist: pytorch Variable, with shape [m, m] / float
    """
    m = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m)
    yy = xx.t()
    dist = xx + yy
    dist.addmm_(x, x.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=0).sqrt()  # for numerical stability
    if(m!=1):
        # dist = torch.div(dist, dist.max().expand(m, n))
        num_line = torch.full([m, m], m*(m-1)).to(device)
        dist = torch.div(dist, num_line)
    dist = torch.sum(dist)
    return dist

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename='/media/store1/sxw/code/tood/test.log',
                    filemode='w')

@mmcv.jit(derivate=True, coderize=True)
def distance_loss(pos_target, pos_decode_bbox_pred, pos_decode_bbox_targets, stride):
    """
    Args:
        pos_target(torch.Tensor): the classes of the positive sampling points in one level(one batch).
                which has shape[num_pos,]
        pos_decode_bbox_pred(torch.Tnesor): type(x1,y1,x2,y2) shape[num_pos, 4]
        stride (int): downsamping rate of the img.
    Returns:
        loss(Tensor): distance loss in the batch of the single level.
    """
    logging.debug('stride is' + str(stride))
    # logging.debug('pos_target is' + str(pos_target))
    grouped_inds_list = pos_idx_groupbyClass(pos_target)
    # logging.debug('grouped_inds_list is' + str(grouped_inds_list))
    res = []
    for inds in grouped_inds_list:
        # logging.debug('pos_decode_bbox_pred[inds] shape is' + str(pos_decode_bbox_pred[inds].size()))
        pred_oneClass = pos_decode_bbox_pred[inds].squeeze(1)
        target_oneClass = pos_decode_bbox_targets[inds].squeeze(1)[0,:]
        target_scale_mean = ((target_oneClass[2] - target_oneClass[0])**2 + \
                            (target_oneClass[3] - target_oneClass[1])**2).sqrt()
        # logging.debug('target_scale_mean is' + str(target_scale_mean))
        pred_oneClass_x1y1 = pred_oneClass[:,:2]
        pred_oneClass_x2y2 = pred_oneClass[:,2:]
        pred_oneClass_center = (pred_oneClass_x1y1 + pred_oneClass_x2y2) / 2
        # logging.debug('pred_oneClass_center is' + str(pred_oneClass_center))
        res.append(euclidean_dist(pred_oneClass_center)/target_scale_mean)
    sigmoid = nn.Sigmoid()
    if(len(grouped_inds_list)!=0):
        # logging.debug('res require_grad is' + str(res.requires_grad))
        # res_dis = torch.mean(res) # bug
        logging.debug('res is' + str(res))
        # for idx in range(1, len(res)):
        #     res[0] = res[0] + res[idx]
        res_dis = res[0]
        del res
        # logging.debug('res_dis_mean require_grad is' + str(res_dis.requires_grad))
        # logging.debug('res_dis_mean is' + str(res_dis))
        res_dis = torch.div(res_dis, stride)
        # logging.debug('res_dis_div require_grad is' + str(res_dis.requires_grad))
        # logging.debug('res_dis_div is' + str(res_dis))
        res_dis = 2 * sigmoid(res_dis) - 1
        # logging.debug('res_dis_sig require_grad is' + str(res_dis.requires_grad))
        # logging.debug('res_dis_sig is' + str(res_dis)) 
        return res_dis
    else:
        return torch.tensor(0.0, dtype=torch.float)
    # logging.debug('img_size is' + str(img_size[0]) + str(img_size[1]))
    # logging.debug('stride is' + str(stride))
    # logging.debug('distance res is' + str(res))
    # logging.debug('distance loss is' + str(torch.mean(torch.tensor(res, dtype=torch.float))))

@LOSSES.register_module()
class DistanceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 loss_weight=1.0):
        """`Distance Loss <self_made>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(DistanceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
    
    def forward(self,
                pos_target,
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                stride):
        """Forward function.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.use_sigmoid:
            loss_dis = self.loss_weight * distance_loss(
                pos_target, 
                pos_decode_bbox_pred, 
                pos_decode_bbox_targets,
                stride)
        else:
            raise NotImplementedError
        return loss_dis