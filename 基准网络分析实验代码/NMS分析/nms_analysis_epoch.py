from cProfile import label
import enum
import json
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cv2 as cv
# 31322 66038 323799
name = "000000323799"
image_prefix = 'H:/dataset/coco2017/val2017/val2017/'
json_path_12 = 'F:/毕业设计/TOOD-masterv2/nms分析/epoch_12/_'+name+'.json'
json_path_9 = 'F:/毕业设计/TOOD-masterv2/nms分析/epoch_9/_'+name+'.json'
json_path_5 = 'F:/毕业设计/TOOD-masterv2/nms分析/epoch_5/_'+name+'.json'
RGB_list = [(155,136,185),(137,127,59) ,(29,134,177) ,(2,19,83) ,(132,184,222), (148,80,150) ,(163,138,209), (192,98,87) ,(146,177,14) ,(237,54,90) ,(100,130,57) ,(255,190,160), (42,175,72) ,(109,230,201), (128,166,99) ,(226,183,214), (43,126,158) ,(27,191,136) ,(122,8,191) ,(92,162,237) ,(117,214,3) ,(90,248,23) ,(68,25,142) ,(231,140,180), (156,227,248), (142,217,73) ,(99,152,94) ,(183,49,235) ,(219,13,35) ,(52,182,236) ,(38,64,216) ,(58,127,143) ,(243,173,80) ,(6,191,197) ,(22,145,67) ,(214,165,135), (244,181,157), (52,96,12) ,(4,27,230) ,(183,222,72) ,(61,140,105) ,(20,192,20) ,(151,184,80) ,(41,43,60) ,(133,30,74) ,(219,185,115), (162,182,87) ,(134,17,36) ,(238,72,121) ,(178,252,156), (74,85,223) ,(71,70,202) ,(72,104,216) ,(88,219,216) ,(23,184,31) ,(3,201,145) ,(206,151,79) ,(0,146,237) ,(87,16,249) ,(36,183,172) ,(227,186,112), (122,97,223) ,(58,27,213) ,(195,43,136) , (107,148,201), (103,6,6) ,(222,124,47) ,(145,236,21) ,(33,205,190) ,(13,52,126) ,(10,243,162) ,(8,85,242) ,(133,223,255), (207,253,31) ,(229,110,139)]
class_map ={0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane', 5:'bus',6:'train', 7:'truck', 8:'boat', 9:'traffic light', 10:'fire hydrant',
11:'stop sign', 12:'parking meter', 13:'bench', 14:'bird', 15:'cat', 16:'dog',17:'horse', 18:'sheep', 19:'cow', 20:'elephant', 21:'bear', 22:'zebra', 23:'giraffe',24:'backpack', 25:'umbrella', 26:'handbag', 27:'tie', 28:'suitcase', 29:'frisbee',30:'skis', 31:'snowboard', 32:'sports ball', 33:'kite', 34:'baseball bat',35:'baseball glove', 36:'skateboard', 37:'surfboard', 38:'tennis racket',39:'bottle', 40:'wine glass', 41:'cup', 42:'fork', 43:'knife', 44:'spoon', 45:'bowl',46:'banana', 47:'apple', 48:'sandwich', 49:'orange', 50:'broccoli', 51:'carrot',52:'hot dog', 53:'pizza', 54:'donut', 55:'cake', 56:'chair', 57:'couch',58:'potted plant', 59:'bed', 60:'dining table', 61:'toilet', 62:'tv', 63:'laptop',64:'mouse', 65:'remote', 66:'keyboard', 67:'cell phone', 68:'microwave',69:'oven', 70:'toaster', 71:'sink', 72:'refrigerator', 73:'book', 74:'clock',75:'vase', 76:'scissors', 77:'teddy bear', 78:'hair drier', 79:'toothbrush'}
thr = 0
thr_score = 0.4

sns.set_theme(style='white')

with open(json_path_12, 'r') as file:
    data12 = json.load(file)
with open(json_path_9, 'r') as file:
    data9 = json.load(file)
with open(json_path_5, 'r') as file:
    data5 = json.load(file)
# image_name = data['image_name']
image_name = name+'.jpg'
# image = cv.imread(image_prefix + image_name)
image = cv.imread(image_prefix + image_name)
origin_bboxes_12 = data12['origin_bbox']
origin_bboxes_9 = data9['origin_bbox']
origin_bboxes_5 = data5['origin_bbox']
# for idx, data in enumerate(origin_bboxes):
#     print(idx, data[4])

origin_bbox12 = [data[:4] for data in origin_bboxes_12]
origin_score12 = [data[4] for data in origin_bboxes_12]
origin_score_thr12 = np.array(origin_score12)[(np.array(origin_score12)>=thr_score).nonzero()].tolist()
origin_label12 = [data[5] for data in origin_bboxes_12]

origin_bbox9 = [data[:4] for data in origin_bboxes_9]
origin_score9 = [data[4] for data in origin_bboxes_9]
origin_score_thr9 = np.array(origin_score9)[(np.array(origin_score9)>=thr_score).nonzero()].tolist()
origin_label9 = [data[5] for data in origin_bboxes_9]

origin_bbox5 = [data[:4] for data in origin_bboxes_5]
origin_score5 = [data[4] for data in origin_bboxes_5]
origin_score_thr5 = np.array(origin_score5)[(np.array(origin_score5)>=thr_score).nonzero()].tolist()
origin_label5 = [data[5] for data in origin_bboxes_5]

print(np.mean(origin_score_thr12), np.mean(origin_score_thr9), np.mean(origin_score_thr5))
print(np.std(origin_score_thr12), np.std(origin_score_thr9), np.std(origin_score_thr5))

origin_bbox_class12, origin_bbox_class9, origin_bbox_class5 = [], [], []
origin_score_class12, origin_score_class9, origin_score_class5 = [], [], []

for idx in np.unique(origin_label12):
    # print(idx)
    mask = ((np.array(origin_label12 == idx)) & (np.array(origin_score12) > thr)).nonzero()
    origin_bbox_class12.append(np.array(origin_bbox12)[mask].tolist())
    origin_score_class12.append(np.array(origin_score12)[mask].tolist())

for idx in np.unique(origin_label9):
    # print(idx)
    mask = ((np.array(origin_label9 == idx)) & (np.array(origin_score9) > thr)).nonzero()
    origin_bbox_class9.append(np.array(origin_bbox9)[mask].tolist())
    origin_score_class9.append(np.array(origin_score9)[mask].tolist())

for idx in np.unique(origin_label5):
    # print(idx)
    mask = ((np.array(origin_label5 == idx)) & (np.array(origin_score5) > thr)).nonzero()
    origin_bbox_class5.append(np.array(origin_bbox5)[mask].tolist())
    origin_score_class5.append(np.array(origin_score5)[mask].tolist())

nms_bboxes_12 = data12['nms_bbox']
nms_bboxes_9 = data9['nms_bbox']
nms_bboxes_5 = data5['nms_bbox']

nms_bbox12 = [data[:4] for data in nms_bboxes_12]
nms_score12 = [data[4] for data in nms_bboxes_12]
nms_score_thr12 = np.array(nms_score12)[(np.array(nms_score12)>=thr_score).nonzero()].tolist()
nms_label12 = [data[5] for data in nms_bboxes_12]

nms_bbox9 = [data[:4] for data in nms_bboxes_9]
nms_score9 = [data[4] for data in nms_bboxes_9]
nms_score_thr9 = np.array(nms_score9)[(np.array(nms_score9)>=thr_score).nonzero()].tolist()
nms_label9 = [data[5] for data in nms_bboxes_9]

nms_bbox5 = [data[:4] for data in nms_bboxes_5]
nms_score5 = [data[4] for data in nms_bboxes_5]
nms_score_thr5 = np.array(nms_score5)[(np.array(nms_score5)>=thr_score).nonzero()].tolist()
nms_label5 = [data[5] for data in nms_bboxes_5]

nms_bbox_class12, nms_bbox_class9, nms_bbox_class5 = [], [], []
nms_score_class12, nms_score_class9, nms_score_class5 = [], [], []

for idx in np.unique(nms_label12):
    mask = ((np.array(nms_label12) == idx) & (np.array(nms_score12) > thr)).nonzero()
    nms_bbox_class12.append(np.array(nms_bbox12)[mask].tolist())
    nms_score_class12.append(np.array(nms_score12)[mask].tolist())

for idx in np.unique(nms_label9):
    mask = ((np.array(nms_label9) == idx) & (np.array(nms_score9) > thr)).nonzero()
    nms_bbox_class9.append(np.array(nms_bbox9)[mask].tolist())
    nms_score_class9.append(np.array(nms_score9)[mask].tolist())

for idx in np.unique(nms_label5):
    mask = ((np.array(nms_label5) == idx) & (np.array(nms_score5) > thr)).nonzero()
    nms_bbox_class5.append(np.array(nms_bbox5)[mask].tolist())
    nms_score_class5.append(np.array(nms_score5)[mask].tolist())

f, (ax_box1, ax_box2, ax_box3, ax_hist) = plt.subplots(4, sharex=True, gridspec_kw={"height_ratios": (.1, .1, .1, .7)})
sns.boxplot(origin_score_thr12, ax=ax_box3,color='yellowgreen')
sns.boxplot(origin_score_thr9, ax=ax_box2,color='goldenrod')
sns.boxplot(origin_score_thr5, ax=ax_box1,color='salmon')
sns.histplot(origin_score_thr12,kde=True,stat='count',bins=30,color='yellowgreen',label='Epoch_12')
sns.histplot(origin_score_thr9,kde=True,stat='count',bins=30,color='goldenrod',label='Epoch_9')
sns.histplot(origin_score_thr5,kde=True,stat='count',bins=30,color='salmon',label='Epoch_5')
# sns.histplot(data=df_type, x='score',kde=True,stat='probability',bins=20,hue='type',multiple='layer')
plt.xlabel('Epoch 5/9/12 bbox')
plt.legend()
plt.show()

# for idx1, rect_class in enumerate(nms_bbox_class12):
#     temp1 = np.zeros(image.shape, np.uint8)
#     for _, rect in enumerate(rect_class):
#         temp1 = cv.rectangle(temp1, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), RGB_list[idx1], 1)
#     image = cv.addWeighted(image, 1, temp1, 1, 1)
# # for rect in nms_bbox_class[0]:
# #     image = cv.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 0)
# cv.imwrite('F:/nms_temp/'+name+'_test.jpg',image)
# cv.imshow('1',image)
# cv.waitKey(0)