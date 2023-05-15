import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

json_name = "F:/毕业设计/TOOD-masterv2/score_iou分析/score_iou_epoch12.json"

with open(json_name, 'r') as file:
    data = json.load(file)

score_list, iou_list = [], []

for image in data:
    score_list = score_list + image["score"]
    iou_list = iou_list + image["iou"]

score_list_class = score_list.copy()

for idx, _ in enumerate(score_list):
    score_list_class[idx] = score_list[idx] /(iou_list[idx]**6)
low, mid1, mid2, high = 0, 0, 0, 0
for idx, _ in enumerate(score_list_class):
    if(score_list_class[idx]<0.4):
        low += 1
    elif(score_list_class[idx]>=0.4 and score_list_class[idx]<0.6):
        mid1 += 1
    elif(score_list_class[idx]>=0.6 and score_list_class[idx]<0.8):
        mid2 += 1
    else:
        high += 1

low, mid1, mid2, high = low/len(score_list_class), mid1/len(score_list_class), mid2/len(score_list_class), high/len(score_list_class)

print(low, mid1, mid2, high)

df1 = pd.DataFrame({'一致性指标':score_list, "交并比":iou_list})
df2 = pd.DataFrame({'一致性指标':score_list_class, "交并比":iou_list})
print(df2.corr("spearman"))
sns.set(font="SimSun", style="white")
sns.scatterplot(x='一致性指标', y='交并比', data=df1)

plt.show()
plt.waitKey(0)
