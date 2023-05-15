import enum
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
import json
import random

# 2261 7784 9448 22935 75456 201072 570169

name = "000000075456"
image_prefix = 'H:/dataset/coco2017/val2017/val2017/'
json_path13 = 'F:/毕业设计/TOOD-masterv2/topk分析/topk13/_'+name+'.json'
json_path32 = 'F:/毕业设计/TOOD-masterv2/topk分析/topk32/_'+name+'.json'
json_path64 = 'F:/毕业设计/TOOD-masterv2/topk分析/topk64/_'+name+'.json'
json_path128 = 'F:/毕业设计/TOOD-masterv2/topk分析/topk128/_'+name+'.json'
json_path256 = 'F:/毕业设计/TOOD-masterv2/topk分析/topk256/_'+name+'.json'
RGB_list = [(155,136,185),(137,127,59) ,(29,134,177) ,(2,19,83) ,(132,184,222), (148,80,150) ,(163,138,209), (192,98,87) ,(146,177,14) ,(237,54,90) ,(100,130,57) ,(255,190,160), (42,175,72) ,(109,230,201), (128,166,99) ,(226,183,214), (43,126,158) ,(27,191,136) ,(122,8,191) ,(92,162,237) ,(117,214,3) ,(90,248,23) ,(68,25,142) ,(231,140,180), (156,227,248), (142,217,73) ,(99,152,94) ,(183,49,235) ,(219,13,35) ,(52,182,236) ,(38,64,216) ,(58,127,143) ,(243,173,80) ,(6,191,197) ,(22,145,67) ,(214,165,135), (244,181,157), (52,96,12) ,(4,27,230) ,(183,222,72) ,(61,140,105) ,(20,192,20) ,(151,184,80) ,(41,43,60) ,(133,30,74) ,(219,185,115), (162,182,87) ,(134,17,36) ,(238,72,121) ,(178,252,156), (74,85,223) ,(71,70,202) ,(72,104,216) ,(88,219,216) ,(23,184,31) ,(3,201,145) ,(206,151,79) ,(0,146,237) ,(87,16,249) ,(36,183,172) ,(227,186,112), (122,97,223) ,(58,27,213) ,(195,43,136) , (107,148,201), (103,6,6) ,(222,124,47) ,(145,236,21) ,(33,205,190) ,(13,52,126) ,(10,243,162) ,(8,85,242) ,(133,223,255), (207,253,31) ,(229,110,139)]
class_map ={0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane', 5:'bus',6:'train', 7:'truck', 8:'boat', 9:'traffic light', 10:'fire hydrant',
11:'stop sign', 12:'parking meter', 13:'bench', 14:'bird', 15:'cat', 16:'dog',17:'horse', 18:'sheep', 19:'cow', 20:'elephant', 21:'bear', 22:'zebra', 23:'giraffe',24:'backpack', 25:'umbrella', 26:'handbag', 27:'tie', 28:'suitcase', 29:'frisbee',30:'skis', 31:'snowboard', 32:'sports ball', 33:'kite', 34:'baseball bat',35:'baseball glove', 36:'skateboard', 37:'surfboard', 38:'tennis racket',39:'bottle', 40:'wine glass', 41:'cup', 42:'fork', 43:'knife', 44:'spoon', 45:'bowl',46:'banana', 47:'apple', 48:'sandwich', 49:'orange', 50:'broccoli', 51:'carrot',52:'hot dog', 53:'pizza', 54:'donut', 55:'cake', 56:'chair', 57:'couch',58:'potted plant', 59:'bed', 60:'dining table', 61:'toilet', 62:'tv', 63:'laptop',64:'mouse', 65:'remote', 66:'keyboard', 67:'cell phone', 68:'microwave',69:'oven', 70:'toaster', 71:'sink', 72:'refrigerator', 73:'book', 74:'clock',75:'vase', 76:'scissors', 77:'teddy bear', 78:'hair drier', 79:'toothbrush'}

with open(json_path13, 'r') as file:
    data13 = json.load(file)
with open(json_path32, 'r') as file:
    data32 = json.load(file)
with open(json_path64, 'r') as file:
    data64 = json.load(file)
with open(json_path128, 'r') as file:
    data128 = json.load(file)
with open(json_path256, 'r') as file:
    data256 = json.load(file)

image_name = name+'.jpg'
image = cv.imread(image_prefix + image_name)
print(image.shape)
gt_bboxes = data13['gt_bbox']
semantic_center = [[320.0, 224.0]]
for c in semantic_center:
    c[0] = image.shape[1] - c[0]
rgb_random = []
for idx, gt_bbox in enumerate(gt_bboxes):
    rgb_random.append(RGB_list[random.randint(0,79)])
    image = cv.rectangle(image, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), rgb_random[-1], 2)
    cv.circle(image, tuple(map(int, semantic_center[idx])), 12, (0,255,0), 2)

pos_anchors = data13['pos_anchor']
pos_labels = data13['pos_anchor_label']
pos_anchors_class = [[] for _ in range(len(gt_bboxes))]
for idx, pos_anchor in enumerate(pos_anchors):
    pos_anchors_class[pos_labels[idx]].append(pos_anchor)
for idx, pos_oneclass in enumerate(pos_anchors_class):
    for pos in pos_oneclass:
        cv.circle(image, tuple(map(int, pos)), 4, rgb_random[idx], -1)

cv.imshow("1", image)
cv.waitKey(0)
# cv.imwrite("F:/topk_temp/num_analysis/"+name+"_topk13.jpg",image)
