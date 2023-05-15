import enum
import json
from operator import gt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

log_path = "F:/毕业设计/TOOD-masterv2/topk分析/topk13.log"
metric_list = []
label_list = []
gt_list = []
anchor_list = []
with open(log_path, 'r') as file:
    while True:
        lines = file.readline()
        if not lines:
            break
        if("norm alignment metrics" in lines):
            st = 0
            for idx, c in enumerate(lines):
                if(c=='['):
                    st = idx
                    break
            metric_list.append(json.loads(lines[st:]))
        elif("pos anchors label" in lines):
            st = 0
            for idx, c in enumerate(lines):
                if(c=='['):
                    st = idx
                    break
            label_list.append(json.loads(lines[st:]))
        elif("gt bbox" in lines):
            st = 0
            for idx, c in enumerate(lines):
                if(c=='['):
                    st = idx
                    break
            gt_list.append(json.loads(lines[st:]))
        elif("pos anchors is" in lines):
            st = 0
            for idx, c in enumerate(lines):
                if(c=='['):
                    st = idx
                    break
            anchor_list.append(json.loads(lines[st:]))

metric_sum, metric_num, centerness_total = 0, 0, 0

for idx in range(len(gt_list)):
    metric_sum += np.sum(metric_list[idx])
    metric_num += len(metric_list[idx])
    centerness_gt = 0
    for gt_inds in np.unique(label_list[idx]):
        mask = (np.array(label_list[idx]) == gt_inds).nonzero()
        anchor_list_oneclass = np.array(anchor_list[idx])[mask].tolist()
        gt_bbox = gt_list[idx][gt_inds]
        centerness = 0
        for anchor in anchor_list_oneclass:
            l = anchor[0] - gt_bbox[0]
            r = gt_bbox[2] - anchor[0]
            t = anchor[1] - gt_bbox[1]
            b = gt_bbox[3] - anchor[1]
            centerness += np.sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
        centerness /= len(anchor_list_oneclass)
        centerness_gt += centerness
    centerness_gt /= len(np.unique(label_list[idx]))
    centerness_total += centerness_gt
metric_mean = metric_sum / metric_num
centerness_mean = centerness_total / len(gt_list)

print(metric_mean, centerness_mean)


