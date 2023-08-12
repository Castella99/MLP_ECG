import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, average_precision_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay, auc, RocCurveDisplay
import torch.optim as optim
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Dataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.manifold import TSNE
import random
import math
import copy
from tqdm import tqdm

def get_clf_eval(y_test, y_pred, y_pred_rate):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0, average='macro')
    recall = recall_score(y_test, y_pred, zero_division=0, average='macro')
    F1 = f1_score(y_test, y_pred, zero_division=0, average='macro')
    try :
        AUROC = roc_auc_score(y_test, y_pred_rate, average='macro')
    except :
        AUROC = 0
    AUPRC = average_precision_score(y_test, y_pred_rate, average='macro')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    return [accuracy, precision, recall, F1, AUROC, AUPRC, specificity]

def get_eval(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0, average='macro')
    recall = recall_score(y_test, y_pred, zero_division=0, average='macro')
    F1 = f1_score(y_test, y_pred, zero_division=0, average='macro')
    try :
        AUROC = roc_auc_score(y_test, y_pred, average='macro')
    except :
        AUROC = 0
    AUPRC = average_precision_score(y_test, y_pred, average='macro')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    return [accuracy, precision, recall, F1, AUROC, AUPRC, specificity]

def get_class_weights(label_dataset):
    ## Class 가중치 생성
    neg, pos = np.bincount(label_dataset.reshape(-1))
    total = neg + pos
    
    #tf way
    weight_for_0 = (1 / neg) * (total/2)
    weight_for_1 = (1 / pos) * (total/2)
            
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    return class_weight
    
def make_dataloader(x_train, y_train, x_val, y_val, x_test, y_test, x_fine=None, y_fine=None, unlabeled=None, batch_size=16) :
    # make data loader
    print("make data loader\n")
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    if unlabeled != None :
        unlabeled_dataset = TensorDataset(torch.from_numpy(unlabeled))
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size*8, shuffle=True)
        finetune_dataset = TensorDataset(torch.from_numpy(x_fine), torch.from_numpy(y_fine))
        finetune_dataloader = DataLoader(finetune_dataset, batch_size, shuffle=True)
        return train_dataloader, finetune_dataloader, val_dataloader, test_dataloader, unlabeled_dataloader
    return train_dataloader, val_dataloader, test_dataloader

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.name}_finetune'
    else:
        name = args.name
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)
        
def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
class StratifiedSampler(Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

def plot_features(features, labels, num_classes, epoch=0, prefix="", save=""):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    labels = np.argmax(labels, axis=-1)
    tsne = TSNE(2)
    features = tsne.fit_transform(features)
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.title(f"{prefix} {epoch+1} Epoch Feature Plot")
    plt.legend(['0', '1'], loc='upper right')
    plt.show()
    if save != "" :
        plt.save(f"{save}_{prefix}_{epoch+1}.png")
    plt.close()
    
def make_TSNE(model, test_loader, save="") :
    model.dense = nn.Identity()
    dataloader = DataLoader(test_loader.dataset, batch_size=1)
    tsne = TSNE(2)
    
    model.eval()
    with torch.no_grad():
        true_list = []
        false_list = []
        for step, (ecg, targets) in enumerate(dataloader) :
            batch_size = ecg.shape[0]
            ecg = ecg.cuda()
            outputs = model(ecg)
            if targets.item() == 1. :
                true_list.append(outputs.detach().cpu().numpy().reshape(-1))
            else :
                false_list.append(outputs.detach().cpu().numpy().reshape(-1))
        EMB_true = np.stack(true_list)
        EMB_false = np.stack(false_list)
    
    EMB_true = tsne.fit_transform(EMB_true)
    EMB_false = tsne.fit_transform(EMB_false)
    
    plt.figure(figsize=(10,10), dpi=300)
    plt.scatter(EMB_true[:,0], EMB_true[:,1], label="True", s=1)
    plt.scatter(EMB_false[:,0], EMB_false[:,1], label="False", s=1)
    plt.legend()
    plt.title("T-SNE Embedding")
    plt.show()
            
    if save != "" :
        plt.savefig(save)