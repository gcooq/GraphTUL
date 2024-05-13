from __future__ import division
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
def accuracy_K(pred,label):
    Total=len(pred)
    acc1=0
    acc5=0
    for index in range(Total):
        top1 = np.argpartition(a=-pred[index], kth=1)[:1]
        top5 = np.argpartition(a=-pred[index], kth=5)[:5]
        if label[index] in top1:
            acc1+= 1
        if label[index] in top5:
            acc5+= 1
    acc1_rate=acc1/Total
    acc5_rate=acc5/Total
    return acc1_rate,acc5_rate
def macro_F(pred,label):
    pred_label=[]
    Total = len(pred)
    for i in range(Total):
        pred_label.append(np.argmax(pred[i]))
    #macro_f=f1_score(label[:Total],pred_label,average='macro') cannot use this for macro-f1 immergence!!!!!!
    macro_r=recall_score(label[:Total],pred_label,average='macro')
    macro_p=precision_score(label[:Total],pred_label,average='macro')
    macro_f=2*macro_p*macro_r/(macro_p+macro_r)
    return macro_f,macro_r,macro_p