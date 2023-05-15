import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

json_prefix = 'F:/毕业设计/TOOD-masterv2/框质量分析/'
json_name = '20image_5.json'

with open(json_prefix+json_name, 'r') as file:
    data = json.load(file)

pre_score, pre_iou, aft_score, aft_iou = [], [], [], []
pre_score_list, pre_iou_list, aft_score_list, aft_iou_list = [], [], [], []
for image in data:
    pre_score.append(np.array(image['score1']).reshape(-1).tolist())
    aft_score.append(np.array(image['score2']).reshape(-1).tolist())
    pre_iou.append(np.array(image['iou1']).reshape(-1).tolist())
    aft_iou.append(np.array(image['iou2']).reshape(-1).tolist())

for idx in range(len(pre_score)):
    pre_score_list.extend(pre_score[idx])
    aft_score_list.extend(aft_score[idx])
    pre_iou_list.extend(pre_iou[idx])
    aft_iou_list.extend(aft_iou[idx])

pre_score_list_nozero, pre_iou_list_nozero, aft_score_list_nozero, aft_iou_list_nozero = [], [], [], []

for idx in range(len(aft_score_list)-1):
    if(not(pre_score_list[idx]==0 or aft_score_list[idx]==0)):
        pre_score_list_nozero.append(pre_score_list[idx])
        aft_score_list_nozero.append(aft_score_list[idx])
    if(not(pre_iou_list[idx]==0 or aft_iou_list[idx]==0)):
        pre_iou_list_nozero.append(pre_iou_list[idx])
        aft_iou_list_nozero.append(aft_iou_list[idx])

df_pre = pd.DataFrame({'分类分数':pre_score_list, '交并比':pre_iou_list})
df_aft = pd.DataFrame({'分类分数':aft_score_list, '交并比':aft_iou_list})
df_iou = pd.DataFrame({'第一次回归':pre_iou_list_nozero, '第二次回归':aft_iou_list_nozero})
df_score = pd.DataFrame({'第一次回归':pre_score_list_nozero, '第二次回归':aft_score_list_nozero})

sns.set(font_scale=1.1, style='white', font='SimSun')

# sns.scatterplot(x='分类分数', y='交并比', data=df_aft)
sns.scatterplot(x='第一次回归', y='第二次回归', data=df_iou)

plt.show()

