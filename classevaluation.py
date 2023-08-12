import numpy as np
import os
from functools import reduce 

## (1) Muticlass 유효성 검사 함수 : get_multiclass_evaluation
# input  : i_class_confusion_matrix
# output : i_class_evaluation list [c_no, i_accuracy,i_precision,i_recall,i_specificity,i_f1_score,i_ppv,i_npv,i_class_cnt]

def get_class_evaluation(i_class_confusion_matrix):
    c_cm   = np.array(i_class_confusion_matrix)
    c_len  = c_cm.shape[0]

    i_class_evaluation=[]
    for c_no in  range(c_len):
        # [tn, fp, 
        #  fn, tp]
        (i_tn,i_fp,i_fn,i_tp,i_er,i_cnt,i_acc_cnt, i_class_cnt)=(0,0,0,0,0,0,0,0)
        for t in range(c_len):
            for p in range(c_len):
                #print("c[t][p]",i_cm[t][p])
                if   c_no==t and c_no==p: 
                     i_tp = i_tp+int(c_cm[t][p]) 
                     i_class_cnt=i_class_cnt+int(c_cm[t][p])
                elif c_no==t and c_no!=p: 
                     i_fn = i_fn+int(c_cm[t][p]) 
                     i_class_cnt=i_class_cnt+int(c_cm[t][p])
                elif c_no!=t and c_no==p: 
                     i_fp = i_fp+int(c_cm[t][p])
                elif c_no!=t and c_no!=p: 
                     i_tn = i_tn+int(c_cm[t][p])
                else:               
                     i_er = i_er+1
                if   t == p:
                     i_acc_cnt = i_acc_cnt+int(c_cm[t][p])
                i_cnt=i_cnt+int(c_cm[t][p])
        ###  Accuracy, Precision, Recall, Specificity, f1_score,Positive Predictive Value,Negative Predictive Value
        # 1) Accuracy(정확도) 
        #if (i_tp+i_fn+i_fp+i_tn)==0:     i_accuracy    = 0.0 
        #else:                            i_accuracy    = float((i_tp+i_tn)/(i_tp+i_fn+i_fp+i_tn))
        if  i_cnt == 0:                   i_accuracy    = 0.0 
        else:                             i_accuracy    = float(i_acc_cnt/i_cnt)
        # 2) Precision(정밀도)  =  양성예측도(Positive Predictive Value)
        if (i_tp+i_fp)==0:               i_precision   = 0.0 
        else:                            i_precision   = float((i_tp)/(i_tp+i_fp))
        # 3) Recall(재현율)     = Sensitivity (민감도)
        if (i_tp+i_fn)==0:               i_recall      = 0.0 
        else:                            i_recall      = float((i_tp)/(i_tp+i_fn))
        # 4) Specificity(특이도)
        if (i_fp+i_tn)==0:               i_specificity = 0.0 
        else:                            i_specificity = float((i_tn)/(i_fp+i_tn))
        # 5) f1_score
        if (i_precision==0 or i_recall==0): i_f1_score    = 0.0
        else:                               i_f1_score    = float(2*(1/((1/i_precision)+(1/i_recall))))
        # 6) 양성예측도(Positive Predictive Value) = Precision
        if (i_tp+i_fp)==0:               i_ppv       = 0.0
        else:                            i_ppv       = float((i_tp)/(i_tp+i_fp))               
        if (i_tn+i_fn)==0:               i_npv       = 0.0
        # 7) 음성예측도(Negative Predictive Value)    
        else:                            i_npv       = float((i_tn)/(i_tn+i_fn))  
        i_class_evaluation.append([c_no, i_accuracy,i_precision,i_recall,i_specificity,i_f1_score,i_ppv,i_npv,i_class_cnt])    
        #print(t_class_total)
    return i_class_evaluation

## (2) Muticlass 유효성 검사 함수 : get_multiclass_evaluation
# input  : i_multiclass_confusion_matrix
# output : i_class_evaluation list [c_no, i_accuracy,i_precision,i_recall,i_specificity,i_f1_score,i_ppv,i_npv,i_class_cnt]

def get_multiclass_evaluation(i_multiclass_confusion_matrix):
    c_cm   = np.array(i_multiclass_confusion_matrix)
    c_len  = c_cm.shape[0]

    i_class_evaluation=[]
    (c_accuracy,c_precision,c_recall,c_specificity,c_f1_score,c_ppv,c_npv,c_class_cnt)=(0,0,0,0,0,0,0,0)
    (c_w_accuracy,c_w_precision,c_w_recall,c_w_specificity,c_w_f1_score,c_w_ppv,c_w_npv,c_w_class_cnt)=(0,0,0,0,0,0,0,0)
    for c_no in  range(c_len):
        # [tn, fp, 
        #  fn, tp]
        (i_tn,i_fp,i_fn,i_tp,i_er,i_cnt,i_acc_cnt, i_class_cnt)=(0,0,0,0,0,0,0,0)
        for t in range(c_len):
            for p in range(c_len):
                #print("c[t][p]",i_cm[t][p])
                if   c_no==t and c_no==p: 
                     i_tp = i_tp+int(c_cm[t][p]) 
                     i_class_cnt=i_class_cnt+int(c_cm[t][p])
                elif c_no==t and c_no!=p: 
                     i_fn = i_fn+int(c_cm[t][p]) 
                     i_class_cnt=i_class_cnt+int(c_cm[t][p])
                elif c_no!=t and c_no==p: 
                     i_fp = i_fp+int(c_cm[t][p])
                elif c_no!=t and c_no!=p: 
                     i_tn = i_tn+int(c_cm[t][p])
                else:               
                     i_er = i_er+1
                if   t == p:
                     i_acc_cnt = i_acc_cnt+int(c_cm[t][p])
                i_cnt=i_cnt+int(c_cm[t][p])
        ###  Accuracy, Precision, Recall, Specificity, f1_score,Positive Predictive Value,Negative Predictive Value
        # 1) Accuracy(정확도) 
        #if (i_tp+i_fn+i_fp+i_tn)==0:     i_accuracy    = 0.0 
        #else:                            i_accuracy    = float((i_tp+i_tn)/(i_tp+i_fn+i_fp+i_tn))
        if  i_cnt == 0:                   i_accuracy    = 0.0 
        else:                             i_accuracy    = float(i_acc_cnt/i_cnt)
        # 2) Precision(정밀도)  =  양성예측도(Positive Predictive Value)
        if (i_tp+i_fp)==0:               i_precision   = 0.0 
        else:                            i_precision   = float((i_tp)/(i_tp+i_fp))
        # 3) Recall(재현율)     = Sensitivity (민감도)
        if (i_tp+i_fn)==0:               i_recall      = 0.0 
        else:                            i_recall      = float((i_tp)/(i_tp+i_fn))
        # 4) Specificity(특이도)
        if (i_fp+i_tn)==0:               i_specificity = 0.0 
        else:                            i_specificity = float((i_tn)/(i_fp+i_tn))
        # 5) f1_score
        if (i_precision==0 or i_recall==0): i_f1_score    = 0.0
        else:                               i_f1_score    = float(2*(1/((1/i_precision)+(1/i_recall))))
        # 6) 양성예측도(Positive Predictive Value) = Precision
        if (i_tp+i_fp)==0:               i_ppv       = 0.0
        else:                            i_ppv       = float((i_tp)/(i_tp+i_fp))               
        if (i_tn+i_fn)==0:               i_npv       = 0.0
        # 7) 음성예측도(Negative Predictive Value)    
        else:                            i_npv       = float((i_tn)/(i_tn+i_fn))  
        i_class_evaluation.append([c_no, i_accuracy,i_precision,i_recall,i_specificity,i_f1_score,i_ppv,i_npv,i_class_cnt])    
        #print(t_class_total)

        c_accuracy    = c_accuracy + i_accuracy
        c_precision   = c_precision + i_precision
        c_recall      = c_recall + i_recall
        c_specificity = c_specificity + i_specificity
        c_f1_score    = c_f1_score + i_f1_score
        c_ppv         = c_ppv + i_ppv
        c_npv         = c_npv + i_npv
        c_class_cnt   = c_class_cnt + i_class_cnt
        
        c_w_accuracy    = c_w_accuracy + i_accuracy*i_class_cnt
        c_w_precision   = c_w_precision + i_precision*i_class_cnt
        c_w_recall      = c_w_recall + i_recall*i_class_cnt
        c_w_specificity = c_w_specificity + i_specificity*i_class_cnt
        c_w_f1_score    = c_w_f1_score + i_f1_score*i_class_cnt
        c_w_ppv         = c_w_ppv + i_ppv*i_class_cnt
        c_w_npv         = c_w_npv + i_npv*i_class_cnt
        c_w_class_cnt   = c_w_class_cnt + i_class_cnt        
        
    # Macro Average    
    c_no='macro_avg'
    i=0
    i_class_evaluation.append([c_no, c_accuracy/c_len,c_precision/c_len,c_recall/c_len,c_specificity/c_len,c_f1_score/c_len,
                               c_ppv/c_len,c_npv/c_len,c_class_cnt])          
    # Macro Average    
    c_no='weighted_avg'
    i=0
    i_class_evaluation.append([c_no, c_w_accuracy/c_w_class_cnt,c_w_precision/c_w_class_cnt,c_w_recall/c_w_class_cnt,
                               c_w_specificity/c_w_class_cnt,c_w_f1_score/c_w_class_cnt,c_w_ppv/c_w_class_cnt,c_w_npv/c_w_class_cnt,
                               c_w_class_cnt])             
        
    return i_class_evaluation

### 3) MutiLabel 유효성 검사 함수 : get_mutilabel_evaluation  
# input  : i_multiclass_confusion_matrix
# output : i_class_evaluation list [c_no, i_accuracy,i_precision,i_recall,i_specificity,i_f1_score,i_ppv,i_npv,i_class_cnt]

def get_multilabel_evaluation(i_multilabel_confusion_matrix):
    # cm.shape = [c,2,2]
    c_cm   = np.array(i_multilabel_confusion_matrix)
    c_len  = c_cm.shape[0]

    # tm.shape = [2,2]
    t_cm         = reduce(lambda x, y:x+y,c_cm)

    t_class_evaluation=[]
    (c_accuracy,c_precision,c_recall,c_specificity,c_f1_score,c_ppv,c_npv,c_class_cnt)=(0,0,0,0,0,0,0,0)
    (c_w_accuracy,c_w_precision,c_w_recall,c_w_specificity,c_w_f1_score,c_w_ppv,c_w_npv,c_w_class_cnt)=(0,0,0,0,0,0,0,0)
    for c_no in  range(c_len): 
        i_class_evaluation=[]
        i_cm   = c_cm[c_no,:]
        (i_accuracy,i_precision,i_recall,i_specificity,i_f1_score,i_ppv,i_npv,i_class_cnt)=(0,0,0,0,0,0,0,0)
        for i in  [1]:  ## True = 1
            # [tn, fp, 
            #  fn, tp]
            (i_tn,i_fp,i_fn,i_tp,i_er,i_cnt,i_acc_cnt,i_class_cnt)=(0,0,0,0,0,0,0,0)
            for t in range(2): 
                for p in range(2):
                    #print("c[t][p]",i_cm[t][p])
                    if   i==t and i==p: 
                         i_tp = i_tp+int(i_cm[t][p])
                         i_class_cnt=i_class_cnt+int(i_cm[t][p])
                    elif i==t and i!=p: 
                         i_fn = i_fn+int(i_cm[t][p])
                         i_class_cnt=i_class_cnt+int(i_cm[t][p])
                    elif i!=t and i==p: 
                         i_fp = i_fp+int(i_cm[t][p])
                    elif i!=t and i!=p: 
                         i_tn = i_tn+int(i_cm[t][p])
                    else:               
                         i_er = i_er+1
                    if   t == p:
                         i_acc_cnt = i_acc_cnt+int(i_cm[t][p])
                    i_cnt=i_cnt+int(i_cm[t][p])
            ###  Accuracy, Precision, Recall, Specificity, f1_score,Positive Predictive Value,Negative Predictive Value
            # 1) Accuracy(정확도) 
            #if (c_tp+c_fn+c_fp+c_tn)==0:            c_accuracy  = 0.0 
            #else:                                   c_accuracy  = float((c_tp+c_tn)/(c_tp+c_fn+c_fp+c_tn))
            if i_cnt==0:                            i_accuracy  = 0.0 
            else:                                   i_accuracy  = float((i_acc_cnt)/(i_cnt))
            # 2) Precision(정밀도)  =  양성예측도(Positive Predictive Value) 
            if (i_tp+i_fp)==0:                      i_precision = 0.0 
            else:                                   i_precision = float((i_tp)/(i_tp+i_fp))
            # 3) Recall(재현율)     = Sensitivity (민감도) 
            if (i_tp+i_fn)==0:                      i_recall    = 0.0 
            else:                                   i_recall    = float((i_tp)/(i_tp+i_fn))
            # 4) Specificity(특이도)
            if (i_fp+i_tn)==0:                      i_specificity = 0.0 
            else:                                   i_specificity = float((i_tn)/(i_fp+i_tn))
            # 5) f1_score
            if (i_precision==0 or i_recall==0):     i_f1_score  = 0.0
            else:                                   i_f1_score  = float(2*(1/((1/i_precision)+(1/i_recall))))
            # 6) 양성예측도(Positive Predictive Value) = Precision
            if (i_tp+i_fp)==0:                      i_ppv       = 0.0
            else:                                   i_ppv       = float((i_tp)/(i_tp+i_fp))               
            # 7) 음성예측도(Negative Predictive Value) 
            if (i_tn+i_fn)==0:                      i_npv       = 0.0
            else:                                   i_npv       = float((i_tn)/(i_tn+i_fn))  
            t_class_evaluation.append([c_no, i_accuracy,i_precision,i_recall,i_specificity,i_f1_score,i_ppv,i_npv,i_class_cnt]) 
    
        c_accuracy    = c_accuracy + i_accuracy
        c_precision   = c_precision + i_precision
        c_recall      = c_recall + i_recall
        c_specificity = c_specificity + i_specificity
        c_f1_score    = c_f1_score + i_f1_score
        c_ppv         = c_ppv + i_ppv
        c_npv         = c_npv + i_npv
        c_class_cnt   = c_class_cnt + i_class_cnt
        
        c_w_accuracy    = c_w_accuracy + i_accuracy*i_class_cnt
        c_w_precision   = c_w_precision + i_precision*i_class_cnt
        c_w_recall      = c_w_recall + i_recall*i_class_cnt
        c_w_specificity = c_w_specificity + i_specificity*i_class_cnt
        c_w_f1_score    = c_w_f1_score + i_f1_score*i_class_cnt
        c_w_ppv         = c_w_ppv + i_ppv*i_class_cnt
        c_w_npv         = c_w_npv + i_npv*i_class_cnt
        c_w_class_cnt   = c_w_class_cnt + i_class_cnt
        
    t_tn = t_cm[0,0]
    t_fp = t_cm[0,1]
    t_fn = t_cm[1,0]
    t_tp = t_cm[1,1]
    ###  Accuracy, Precision, Recall, Specificity, f1_score,Positive Predictive Value,Negative Predictive Value
    # 1) Accuracy(정확도) 
    if (t_tp+t_fn+t_fp+t_tn)==0:            t_accuracy  = 0.0 
    else:                                   t_accuracy  = float((t_tp+t_tn)/(t_tp+t_fn+t_fp+t_tn))
    # 2) Precision(정밀도)  =  양성예측도(Positive Predictive Value) 
    if (t_tp+t_fp)==0:                      t_precision = 0.0 
    else:                                   t_precision = float((t_tp)/(t_tp+t_fp))
    # 3) Recall(재현율)     = Sensitivity (민감도) 
    if (t_tp+t_fn)==0:                      t_recall    = 0.0 
    else:                                   t_recall    = float((t_tp)/(t_tp+t_fn))
    # 4) Specificity(특이도)
    if (t_fp+t_tn)==0:                      t_specificity = 0.0 
    else:                                   t_specificity = float((t_tn)/(t_fp+t_tn))
    # 5) f1_score
    if (t_precision==0 or t_recall==0):     t_f1_score  = 0.0
    else:                                   t_f1_score  = float(2*(1/((1/t_precision)+(1/t_recall))))
    # 6) 양성예측도(Positive Predictive Value) = Precision
    if (t_tp+t_fp)==0:                      t_ppv       = 0.0
    else:                                   t_ppv       = float((t_tp)/(t_tp+t_fp))               
    # 7) 음성예측도(Negative Predictive Value) 
    if (t_tn+t_fn)==0:                      t_npv       = 0.0
    else:                                   t_npv       = float((t_tn)/(t_tn+t_fn)) 
    t_class_cnt =  t_tp+t_fn  
 
    # Micro Average    
    c_no='micro_avg'
    t_class_evaluation.append([c_no, t_accuracy,t_precision,t_recall,t_specificity,t_f1_score,t_ppv,t_npv,t_class_cnt])    
    # Macro Average    
    c_no='macro_avg'
    t_class_evaluation.append([c_no, c_accuracy/c_len,c_precision/c_len,c_recall/c_len,c_specificity/c_len,c_f1_score/c_len,
                               c_ppv/c_len,c_npv/c_len,c_class_cnt])          
    # Macro Average    
    c_no='weighted_avg'
    t_class_evaluation.append([c_no, c_w_accuracy/c_w_class_cnt,c_w_precision/c_w_class_cnt,c_w_recall/c_w_class_cnt,
                               c_w_specificity/c_w_class_cnt,c_w_f1_score/c_w_class_cnt,c_w_ppv/c_w_class_cnt,c_w_npv/c_w_class_cnt,
                               c_w_class_cnt])               
    c_no='samples_avg'
    t_class_evaluation.append([c_no, t_recall,t_recall,t_recall,t_recall,t_recall,t_recall,t_recall,c_class_cnt])  
    return t_class_evaluation