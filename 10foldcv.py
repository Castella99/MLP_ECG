import json
import copy
import shutil
import os
import pickle
import numpy as np
import pandas as pd
import gc
import h5py
from configparser import ConfigParser
from models.ECGNET import ResNet1D
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix, precision_score, recall_score, average_precision_score, f1_score, roc_auc_score, roc_curve
import math
from utils import get_clf_eval, get_class_weights, get_cosine_schedule_with_warmup, model_load_state_dict, make_dataloader
from train import train_mpl_loop, finetune, evaluate, train_pl_loop, predict, make_TSNE
from gausrank import GaussRankScaler
import classevaluation as classeval
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

def main(args):
    print("**1. Command Input parameter & Config Parsing")
    # ap config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file, encoding="UTF-8")
    
    # default config
    output_base_dir      = cp["DEFAULT"].get("output_base_dir")
    if args.instance is None:
       instance          = cp["DEFAULT"].get("instance")
    else:
       instance          = args.instance 
    output_dir           = f"{output_base_dir}/{instance}"
    dataset_dir          = cp["DEFAULT"].get("dataset_dir")
    if args.dataset_base_name is None:
       dataset_base_name = cp["DEFAULT"].get("dataset_base_name")
    else:
       dataset_base_name = args.dataset_base_name 

    if args.base_model_name is None:
       base_model_name      = cp["DEFAULT"].get("base_model_name")
    else:
       base_model_name      = args.base_model_name

    if args.class_names is None:
       class_names          = cp["DEFAULT"].get("class_names").split(',')
    else:
        class_names          = args.class_names.split(',')
    
    # train config
    #use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights          = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name       = cp["TRAIN"].get("output_weights_name")

    if args.epochs is None:    
       epochs                 = cp["TRAIN"].getint("epochs")
    else:
       epochs                 = args.epochs 

    if args.batch_size is None:
       batch_size             = cp["TRAIN"].getint("batch_size")
    else:
       batch_size             = args.batch_size
       print("batch size :", batch_size)

    if args.initial_learning_rate is None:    
       initial_learning_rate  = cp["TRAIN"].getfloat("initial_learning_rate")
    else:
       initial_learning_rate  = args.initial_learning_rate
        
    if args.kfold is None:    
       kfold  = 10
    else:
       kfold  = args.kfold

    if args.lead_size is None:
       lead_size             = 5000
    else:
       lead_size             = args.lead_size

    if args.lead_ch is None:
       lead_ch             = 12
    else:
       lead_ch             = args.lead_ch

    generator_workers         = cp["TRAIN"].getint("generator_workers")
    train_steps               = cp["TRAIN"].get("train_steps")
    patience_reduce_lr        = cp["TRAIN"].getint("patience_reduce_lr")
    min_lr                    = cp["TRAIN"].getfloat("min_lr")
    val_steps                 = cp["TRAIN"].get("validation_steps")
    positive_weights_multiply = cp["TRAIN"].getfloat("positive_weights_multiply")

    use_sample_dataset        = cp["TRAIN"].getboolean("use_sample_dataset")
    use_sample_train_count    = cp["TRAIN"].getint("use_sample_train_count")
    use_sample_val_count      = cp["TRAIN"].getint("use_sample_val_count")

    use_fit_generator         = cp["TRAIN"].getboolean("use_fit_generator")
    use_fit_callback          = cp["TRAIN"].getboolean("use_fit_callback")

    dataset_name = f"{dataset_base_name}"
 
    print(f"**2. use dataset : {dataset_name}")

    print(f"**3. use trained model weights : {os.path.isfile(output_dir+'/ecg_cv.csv')}")
    if os.path.isfile(output_dir+'/ecg_cv.csv') :
        # resuming mode
        print("** use trained model weights **")
        # load training status for resuming
        macro_f1_score_list = []
        for dir in os.listdir(output_dir) :
            if os.path.isdir(os.path.join(output_dir, dir)) :
                training_stats_file = os.path.join(output_dir, dir, ".training_stats.json")
                if os.path.isfile(training_stats_file):
                    # TODO: add loading previous learning rate?
                    training_stats = json.load(open(training_stats_file))
                    macro_f1_score_list.append(training_stats["macro_f1"])
                else:
                    training_stats = {}
                    macro_f1_score_list.append(0.0)
        macro_f1_score_np = np.array(macro_f1_score_list)
        idx = np.argmax(macro_f1_score_np)
        dir_name = idx+1
        model_weights_file = os.path.join(output_dir, str(dir_name), f"best_{output_weights_name}")
        print(model_weights_file)
    else :
        model_weights_file = None
        training_stats = None

    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    try:
        print(f"*** Loading HDF5 data file...")
        k = kfold
        dataset_file_path = f"{dataset_dir}/{dataset_name}"
        label_ecg_dataset   = np.array(h5py.File(dataset_file_path, 'r')['full']['ecg'])[:]
        unlabel_ecg_dataset = np.load('./../../data/physionet.org/ecg_arrhythmia.npy').astype('float32')
        label_dataset = np.array(h5py.File(dataset_file_path, 'r')['full']['label'])[:].astype('int')
        
        label_dataset = label_dataset.reshape(-1, 1)
        label_dataset = label_dataset.astype('float32')
        
        print(f"*** Dataset reshape...")
        label_ecg_dataset = label_ecg_dataset[:].reshape(-1, lead_size, lead_ch).astype('float32')
        label_ecg_dataset = np.swapaxes(label_ecg_dataset, 1, 2)
        unlabel_ecg_dataset = np.swapaxes(unlabel_ecg_dataset, 1, 2)
        
        print(f'label_ecg_dataset shape: {label_ecg_dataset.shape}')
        print(f'unlabel_ecg_dataset shape: {unlabel_ecg_dataset.shape}')
        
        print(f"backup config file to {output_dir}")
        shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

        results = []
        myresults =[]
        outputs = []
        val_outputs = []
        
        iter_count = 1
        
        ecg, ecg_test, y_label, y_label_test = train_test_split(label_ecg_dataset, label_dataset, test_size=1/(k+1), stratify=label_dataset, random_state=42)
        ecg_test, ecg_finetune, y_label_test, y_label_finetune = train_test_split(ecg_test, y_label_test, test_size=0.5, stratify=y_label_test, random_state=42)
        
        del label_ecg_dataset, label_dataset
        
        skf = StratifiedKFold(n_splits=2*k, shuffle=True, random_state=42)
        skf_index = list(skf.split(ecg, y_label))
        
        for f in range(k) :
            print(f'{iter_count} fold')           
        
            ecg_train = ecg[skf_index[f][0]]
            y_label_train = y_label[skf_index[f][0]]
            
            ecg_val = ecg[skf_index[f][1]]
            y_label_val = y_label[skf_index[f][1]]
            
            print(f"ecg_train:{ecg_train.shape} ecg_val:{ecg_val.shape} ecg_test:{ecg_test.shape} ecg_finetune:{ecg_finetune.shape}")
               
            train_loader, finetune_loader, val_loader, test_loader, unlabel_loader = \
               make_dataloader(ecg_train, y_label_train, ecg_finetune, y_label_finetune, ecg_val, y_label_val, ecg_test, y_label_test, unlabel_ecg_dataset, args.batch_size)
                           
            #print("** class_weights **")
            #class_weights = get_class_weights(y_label_train)
            #print(class_weights)
            
            model = ResNet1D(12,64,16,4,1,4,1, downsample_gap=1, increasefilter_gap=1).cuda()
            
            summary(model, (12, 5000))
            
            t_model = copy.deepcopy(model)
            s_model = copy.deepcopy(model)
            
            criterion = nn.BCEWithLogitsLoss()
            
            teacher_parameters = [
               {'params': [p for n, p in t_model.named_parameters()]},
            ]
            student_parameters = [
               {'params': [p for n, p in s_model.named_parameters()]},
            ]
            
            t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)
            s_optimizer = optim.SGD(student_parameters,
                                    lr=args.student_lr,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov)
            
            t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
            s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                         args.warmup_steps,
                                                         args.total_steps,
                                                         args.student_wait_steps)
               
            model = train_mpl_loop(args, train_loader, unlabel_loader, val_loader, finetune_loader, 
                       t_model, s_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler)
            
            make_TSNE(copy.deepcopy(model), test_loader, save=f"{output_dir}/{iter_count}Fold_TSNE.png")
            
            y_label_val, y_pred_val = predict(args, model, val_loader)
            y_label_test, y_pred_test = predict(args, model, test_loader)
            
            print("## Threshold Optimization - AUROC")
            fpr, tpr, thresholds = roc_curve(y_label_val, y_pred_val)
            # calculate the g-mean for each threshold
            gmeans = np.sqrt(tpr * (1-fpr))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

            i_threshold=thresholds[ix]
            
            # print('Best Threshold :', best_threshold)
            
            y_true  = y_label_test
            y_pred = (y_pred_test > i_threshold)*1.0
            
            ecg_cv = pd.DataFrame(y_true)
            ecg_cv['ecg_prob'] = y_pred_test
            ecg_cv['ecg_pred'] = y_pred
            
            ecg_val_cv = pd.DataFrame(y_label_val)
            ecg_val_cv['ecg_prob'] = y_pred_val
            
            outputs.append(ecg_cv)
            val_outputs.append(ecg_val_cv)
            
            cm = multilabel_confusion_matrix(y_true, y_pred)
            result = classeval.get_multilabel_evaluation(cm)
            results.append(list(result))
            
            myresult = get_clf_eval(y_true, y_pred, y_pred_test)
            myresults.append(myresult)
            
            print(cm)
            
            print('정확도: {:.3f}'.format(myresult[0]))
            print('정밀도: {:.3f}'.format(myresult[1]))
            print('재현율: {:.3f}'.format(myresult[2]))
            print('F1: {:.3f}'.format(myresult[3]))
            print('AUROC: {:.3f}'.format(myresult[4]))
            print('AUPRC: {:.3f}'.format(myresult[5]))
            print('특이도: {:.3f}'.format(myresult[6]))
            
            model_wts = copy.deepcopy(model.state_dict())
            torch.save(model_wts, f"{output_dir}/{iter_count}Fold_model.pt")
            # print("** done! (End)**")
            iter_count += 1

            del ecg_train
            del ecg_val
            del y_label_train
            del y_label_val
            del y_pred_val
            del y_pred_test
            del y_true
            del y_pred
            del ecg_cv
            del ecg_val_cv
            
            gc.collect()

        # Fold 전체 평균
        list_results = np.array(results)[:,:,1:].astype(np.float64)
        myresults = np.array(myresults)
        myresults_df = pd.DataFrame(myresults, index=[i+1 for i in range(k)], columns=["정확도", "정밀도", "재현율", "F1", "AUROC", "AUPRC", "특이도"])
        myresults_df.to_csv(os.path.join(output_dir, "result.csv"))
        print(f'*******************************************************************************************************')
        print(f'*               All 10-Folding Mean')
        print(f'*******************************************************************************************************')
        print(f'        c_no    accuracy  precision  recall     specificity f1_score    NPV      support')
        print(f'           0    {np.mean(list_results[:,0,0]):.6f}  {np.mean(list_results[:,0,1]):.6f}   {np.mean(list_results[:,0,2]):.6f}   {np.mean(list_results[:,0,3]):.6f}    {np.mean(list_results[:,0,4]):.6f}    {np.mean(list_results[:,0,6]):.6f} {list_results[:,0,7]}')
        print(f'           1    {np.mean(list_results[:,1,0]):.6f}  {np.mean(list_results[:,1,1]):.6f}   {np.mean(list_results[:,1,2]):.6f}   {np.mean(list_results[:,1,3]):.6f}    {np.mean(list_results[:,1,4]):.6f}    {np.mean(list_results[:,1,6]):.6f} {list_results[:,1,7]}')
        print(f'   micro_avg    {np.mean(list_results[:,2,0]):.6f}  {np.mean(list_results[:,2,1]):.6f}   {np.mean(list_results[:,2,2]):.6f}   {np.mean(list_results[:,2,3]):.6f}    {np.mean(list_results[:,2,4]):.6f}    {np.mean(list_results[:,2,6]):.6f} {list_results[:,2,7]}')
        print(f'   macro_avg    {np.mean(list_results[:,3,0]):.6f}  {np.mean(list_results[:,3,1]):.6f}   {np.mean(list_results[:,3,2]):.6f}   {np.mean(list_results[:,3,3]):.6f}    {np.mean(list_results[:,3,4]):.6f}    {np.mean(list_results[:,3,6]):.6f} {list_results[:,3,7]}')
        print(f'weighted_avg    {np.mean(list_results[:,4,0]):.6f}  {np.mean(list_results[:,4,1]):.6f}   {np.mean(list_results[:,4,2]):.6f}   {np.mean(list_results[:,4,3]):.6f}    {np.mean(list_results[:,4,4]):.6f}    {np.mean(list_results[:,4,6]):.6f} {list_results[:,4,7]}')
        print(f'*******************************************************************************************************')
        
        print(f'*******************************************************************************************************')
        print(f'*               All 10-Folding Standard Deviation')
        print(f'*******************************************************************************************************')
        print(f'        c_no    accuracy  precision  recall     specificity f1_score    NPV      support')
        print(f'           0    {np.std(list_results[:,0,0]):.6f}  {np.std(list_results[:,0,1]):.6f}   {np.std(list_results[:,0,2]):.6f}   {np.std(list_results[:,0,3]):.6f}    {np.std(list_results[:,0,4]):.6f}    {np.std(list_results[:,0,6]):.6f} {list_results[:,0,7]}')
        print(f'           1    {np.std(list_results[:,1,0]):.6f}  {np.std(list_results[:,1,1]):.6f}   {np.std(list_results[:,1,2]):.6f}   {np.std(list_results[:,1,3]):.6f}    {np.std(list_results[:,1,4]):.6f}    {np.std(list_results[:,1,6]):.6f} {list_results[:,1,7]}')
        print(f'   micro_avg    {np.std(list_results[:,2,0]):.6f}  {np.std(list_results[:,2,1]):.6f}   {np.std(list_results[:,2,2]):.6f}   {np.std(list_results[:,2,3]):.6f}    {np.std(list_results[:,2,4]):.6f}    {np.std(list_results[:,2,6]):.6f} {list_results[:,2,7]}')
        print(f'   macro_avg    {np.std(list_results[:,3,0]):.6f}  {np.std(list_results[:,3,1]):.6f}   {np.std(list_results[:,3,2]):.6f}   {np.std(list_results[:,3,3]):.6f}    {np.std(list_results[:,3,4]):.6f}    {np.std(list_results[:,3,6]):.6f} {list_results[:,3,7]}')
        print(f'weighted_avg    {np.std(list_results[:,4,0]):.6f}  {np.std(list_results[:,4,1]):.6f}   {np.std(list_results[:,4,2]):.6f}   {np.std(list_results[:,4,3]):.6f}    {np.std(list_results[:,4,4]):.6f}    {np.std(list_results[:,4,6]):.6f} {list_results[:,4,7]}')
        print(f'*******************************************************************************************************')
        
        print(f'*******************************************************************************************************')
        print('정확도: {:.3f}'.format(np.mean(myresults, axis=0)[0]))
        print('정밀도: {:.3f}'.format(np.mean(myresults, axis=0)[1]))
        print('재현율: {:.3f}'.format(np.mean(myresults, axis=0)[2]))
        print('F1: {:.3f}'.format(np.mean(myresults, axis=0)[3]))
        print('AUROC: {:.3f}'.format(np.mean(myresults, axis=0)[4]))
        print('AUPRC: {:.3f}'.format(np.mean(myresults, axis=0)[5]))
        print('특이도: {:.3f}'.format(np.mean(myresults, axis=0)[6]))
        print(f'*******************************************************************************************************')
        print('정확도: {:.3f}'.format(np.std(myresults, axis=0)[0]))
        print('정밀도: {:.3f}'.format(np.std(myresults, axis=0)[1]))
        print('재현율: {:.3f}'.format(np.std(myresults, axis=0)[2]))
        print('F1: {:.3f}'.format(np.std(myresults, axis=0)[3]))
        print('AUROC: {:.3f}'.format(np.std(myresults, axis=0)[4]))
        print('AUPRC: {:.3f}'.format(np.std(myresults, axis=0)[5]))
        print('특이도: {:.3f}'.format(np.std(myresults, axis=0)[6]))

        full_output = pd.concat(outputs)
        val_output = pd.concat(val_outputs)
        full_output.to_csv(output_dir+'/ecg_cv.csv')
        val_output.to_csv(output_dir+'/ecg_val_cv.csv')
        
        print("** done! (End)**")

    finally:
        #os.remove(running_flag_file)
        pass

if __name__ == "__main__":
   ap = argparse.ArgumentParser()
   ap.add_argument('-i', '--instance', type=str, help='instance')
   ap.add_argument('-m', '--base_model_name', type=str, help='base_model_name')
   ap.add_argument('-d', '--dataset_base_name', type=str, help='dataset basename')
   ap.add_argument('-c', '--class_names', type=str, help='class_names')
   
   ap.add_argument('-ep', '--epochs', type=int, help='epoch Number')
   ap.add_argument('-bt', '--batch_size', type=int, help='batch_size')
   ap.add_argument('-lr', '--initial_learning_rate', type=float, help='initial learning rate')    
   ap.add_argument('-kfold', '--kfold', type=int, help='kfold')

   ap.add_argument('-ldsz', '--lead_size', type=int, help='lead size')    
   ap.add_argument('-ldch', '--lead_ch', type=int, help='lead channel count')
   ap.add_argument('--name', type=str, required=True, help='experiment name')
   ap.add_argument('--data-path', default='./data', type=str, help='data path')
   ap.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
   ap.add_argument('--dataset', default='cifar10', type=str,
                     choices=['cifar10', 'cifar100'], help='dataset name')
   ap.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
   ap.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
   ap.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
   ap.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
   ap.add_argument('--start-step', default=0, type=int,
                     help='manual epoch number (useful on restarts)')
   ap.add_argument('--workers', default=4, type=int, help='number of workers')
   ap.add_argument('--num-classes', default=10, type=int, help='number of classes')
   ap.add_argument('--resize', default=32, type=int, help='resize image')
   ap.add_argument('--batch-size', default=16, type=int, help='train batch size')
   ap.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
   ap.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
   ap.add_argument('--teacher_lr', default=0.001, type=float, help='train learning late')
   ap.add_argument('--student_lr', default=0.001, type=float, help='train learning late')
   ap.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
   ap.add_argument('--nesterov', action='store_true', help='use nesterov')
   ap.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
   ap.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
   ap.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
   ap.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
   ap.add_argument('--resume', default='', type=str, help='path to checkpoint')
   ap.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
   ap.add_argument('--finetune', action='store_true',
                     help='only finetune model on labeled dataset')
   ap.add_argument('--finetune-epochs', default=500, type=int, help='finetune epochs')
   ap.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
   ap.add_argument('--finetune-lr', default=3e-5, type=float, help='finetune learning late')
   ap.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
   ap.add_argument('--finetune-momentum', default=0.9, type=float, help='finetune SGD Momentum')
   ap.add_argument('--seed', default=999, type=int, help='seed for initializing training')
   ap.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
   ap.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
   ap.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
   ap.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
   ap.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")

   args = ap.parse_args()
   print(args)
   
   main(args)
