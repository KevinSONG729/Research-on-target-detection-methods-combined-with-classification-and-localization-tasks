# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
import mmcv

INF = 100000000


@BBOX_ASSIGNERS.register_module()
class TaskAlignedAssignerSemantic(BaseAssigner):
    """Task aligned assigner used in the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.

    Assign a corresponding gt bbox or background to each predicted bbox.
    Each bbox will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (int): number of bbox selected in each level
        iou_calculator (dict): Config dict for iou calculator.
            Default: dict(type='BboxOverlaps2D')
    """

    def __init__(self, topk, iou_calculator=dict(type='BboxOverlaps2D')):
        assert topk >= 1
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               pred_scores,
               decode_bboxes,
               anchors,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               alpha=1,
               beta=6):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)


        Args:
            pred_scores (Tensor): predicted class probability,
                shape(n, num_classes)
            decode_bboxes (Tensor): predicted bounding boxes, shape(n, 4)
            anchors (Tensor): pre-defined anchors, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        anchors = anchors[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), anchors.size(0)
        # compute alignment metric between all bbox and gt
        overlaps = self.iou_calculator(decode_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores[:, gt_labels].detach()
        # assign 0 by default
        assigned_gt_inds = anchors.new_full((num_bboxes, ),
                                            0,
                                            dtype=torch.long)
        assign_metrics = anchors.new_zeros((num_bboxes, ))

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = anchors.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No gt boxes, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = anchors.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            return assign_result

        # select top-k bboxes as candidates for each gt
        alignment_metrics = bbox_scores**alpha * overlaps**beta
        topk = min(self.topk, alignment_metrics.size(0))
        _, candidate_idxs = alignment_metrics.topk(topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics[candidate_idxs,
                                              torch.arange(num_gt)]
        is_pos = candidate_metrics > 0

        # limit the positive sample's center in gt
        anchors_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        anchors_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0

        semantic_centers = self.get_Semantic_center(bbox_scores.reshape(-1, num_gt), 
            anchors_cx.unsqueeze(1), 
            anchors_cy.unsqueeze(1), 
            gt_bboxes)

        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_anchors_cx = anchors_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_anchors_cy = anchors_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_anchors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_anchors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_anchors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_anchors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[max_overlaps != -INF] = alignment_metrics[
            max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]

        if(assign_metrics[max_overlaps != -INF].size(0) > 0):
            semantics_centerness_target = self.centerness_semantic_target(
                torch.cat([anchors_cx.unsqueeze(1)[max_overlaps != -INF], anchors_cy.unsqueeze(1)[max_overlaps != -INF]], dim=1),
                semantic_centers,
                argmax_overlaps[max_overlaps != -INF],
                gt_bboxes)
        
            assign_metrics[max_overlaps != -INF] = assign_metrics[max_overlaps != -INF] *\
                semantics_centerness_target

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        assign_result = AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.assign_metrics = assign_metrics
        return assign_result
    
    @mmcv.jit(derivate=True, coderize=True)
    def get_Semantic_center(self, bbox_scores, anchors_cx, anchors_cy, gt_bboxes):
        """get semantic center points in all gt-bbox of one single image.
        
        Args:
            bbox_scores(Tensor): predicted class probability(only about the related gt)
                shape [num_anchors, num_gts]
            anchors_cx(Tensor), anchors_cy(Tensor): the center x-y coordinate of the pr-
            ior anchors. shape [num_anchors, ]
            gt_bboxes(Tensor): groud-Truth bbox in the meta-image with xyxy format.
                shape [num_gts, 4]
        
        Return:
            semantics_center(Tensor): the semantic center points of each gt in single image.
                shape [num_gts, 2]
        """
        semantic_center_list = []
        for idx in range(gt_bboxes.size(0)):
            gt_bbox_oneClass = gt_bboxes[idx,:]
            anchors_cx_inside = (anchors_cx > (gt_bbox_oneClass[0].item()+0.01)) & (anchors_cx < (gt_bbox_oneClass[2].item()-0.01))
            anchors_cy_inside = (anchors_cy > (gt_bbox_oneClass[1].item()+0.01)) & (anchors_cy < (gt_bbox_oneClass[3].item()-0.01))
            anchors_inside_oneClass = (anchors_cx_inside * anchors_cy_inside).reshape(1,-1)[0].nonzero()
            bbox_scores_inOneClass = bbox_scores[anchors_inside_oneClass,idx].squeeze(1)
            if(bbox_scores_inOneClass.size(0) != 0):
                max_score_inOneClass, max_score_inds = torch.max(bbox_scores_inOneClass, dim=0)
                semantic_center_inds = anchors_inside_oneClass[max_score_inds]
                anchors_center = torch.cat([anchors_cx, anchors_cy], dim=1)
                semantic_center = anchors_center[semantic_center_inds]
                semantic_center_list.append(semantic_center)
            else: # the gt-bbox is too small
                semantic_center = torch.cat([((gt_bbox_oneClass[0]+gt_bbox_oneClass[2])/2).int().reshape(1,1), \
                    ((gt_bbox_oneClass[1]+gt_bbox_oneClass[3])/2).int().reshape(1,1)], dim=1)
                semantic_center_list.append(semantic_center)
        semantics_center = torch.cat(semantic_center_list, dim=0)
        return semantics_center
    
    @mmcv.jit(derivate=True, coderize=True)
    def centerness_semantic_target(self, pos_anchor_points, semantic_centers, \
        assigned_gt_inds, gt_bboxes):
        """Compute semantic centerness targets.

        Args:
            pos_anchor_points (Tensor): BBox center-point of positive bboxes in shape
                (num_pos, 2)
            semantic_centers (Tensor): the semantic center-point of each gt-bbox.
                shape (num_gt, 2)
            assigned_gt_inds (Tensor): the assigned label of each positive bbox.
                shape (num_pos, )
            gt_bboxes (Tensor): the groudTruth bbox in the single image.
                shape (num_gt, 4)
            gt_labels (Tensor): the label according to the gt_bboxes.
                shape (num_gt, )

        Returns:
            Tensor: Semantic Centerness target. shape (num_pos, )
        """
        # only calculate pos centerness targets, otherwise there may be nan
        assert pos_anchor_points.size(0) > 0
        centerness_list = []
        for idx, pos_point in enumerate(pos_anchor_points):
            gt_bbox = gt_bboxes[assigned_gt_inds[idx]]
            l = pos_point[0] - gt_bbox[0]
            t = pos_point[1] - gt_bbox[1]
            r = gt_bbox[2] - pos_point[0]
            b = gt_bbox[3] - pos_point[1]
            semantic_center = semantic_centers[assigned_gt_inds[idx]]
            deltax = semantic_center[0] - (gt_bbox[0] + gt_bbox[2]) / 2
            deltay = semantic_center[1] - (gt_bbox[1] + gt_bbox[3]) / 2
            l = l - deltax + torch.abs(deltax)
            r = r + deltax + torch.abs(deltax)
            t = t - deltay + torch.abs(deltay)
            b = b + deltay + torch.abs(deltay)
            centerness_target = (torch.min(l, r) / torch.max(l, r)) * \
                                (torch.min(t, b) / torch.max(t, b)).reshape(1,1)
            centerness_list.append(centerness_target)
        semantic_centerness_target = torch.cat(centerness_list, dim=0)
        return torch.sqrt(semantic_centerness_target).squeeze(1)

