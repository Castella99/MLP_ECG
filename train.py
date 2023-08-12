import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, average_precision_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay, auc, RocCurveDisplay, precision_recall_curve
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, Dataset
from utils import make_dataloader, get_class_weights, get_clf_eval, AverageMeter, save_checkpoint, model_load_state_dict, get_cosine_schedule_with_warmup, get_lr, get_eval, plot_features
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import logging
import time

def train_mpl_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_loader,
               teacher_model, student_model, criterion_xent,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler) :

    print("Learn with Meta Pseudo Label")
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    s_losses = AverageMeter()
    t_losses = AverageMeter()
    t_losses_l = AverageMeter()
    t_losses_mpl = AverageMeter()
    mean_mask = AverageMeter()
    
    best_f1 = 0
    early_stopping = 20
    patience = 0
    best_wts = copy.deepcopy(student_model.state_dict())
    
    for step in range(args.start_step, args.total_steps):
        if step % 100 == 0:
            pbar = tqdm(range(100))
        
        teacher_model.train()
        student_model.train()
        end = time.time()
        
        t_optimizer.zero_grad()
        s_optimizer.zero_grad()

        try:
            ecg_l, targets = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            ecg_l, targets = next(labeled_iter)

        try:
            ecg_ul = next(unlabeled_iter)[0]
        except:
            unlabeled_iter = iter(unlabeled_loader)
            ecg_ul = next(unlabeled_iter)[0]

        data_time.update(time.time() - end)

        ecg_l = ecg_l.cuda()
        ecg_ul = ecg_ul.cuda()
        targets = targets.cuda()
     
        batch_size = ecg_l.shape[0]
        t_ecg = torch.cat((ecg_l, ecg_ul))
        t_logits = teacher_model(t_ecg)
        t_logits_l = t_logits[:batch_size]
        t_logits_ul = t_logits[batch_size:]
        del t_logits

        t_loss_l = criterion_xent(t_logits_l, targets)

        soft_pseudo_label = torch.sigmoid(t_logits_ul.detach() / args.temperature)
        hard_pseudo_label = torch.where(soft_pseudo_label > 0.5, 1.0, 0.0).reshape(-1,1)
        mask = soft_pseudo_label.ge(0.5).float()

        s_ecg = torch.cat((ecg_l, ecg_ul))
        s_logits = student_model(s_ecg)
        s_logits_l = s_logits[:batch_size]
        s_logits_ul = s_logits[batch_size:]
        del s_logits

        s_loss_l_old = F.binary_cross_entropy_with_logits(s_logits_l.detach(), targets)
        
        s_loss_ul = criterion_xent(s_logits_ul, hard_pseudo_label)
        
        s_loss = s_loss_ul
        
        s_loss.backward()
        
        s_optimizer.step()
        s_scheduler.step()

        with torch.no_grad():
            s_logits_l = student_model(ecg_l)
        s_loss_l_new = F.binary_cross_entropy_with_logits(s_logits_l.detach(), targets)

        # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
        # dot_product = s_loss_l_old - s_loss_l_new

        # author's code formula
        dot_product = s_loss_l_new - s_loss_l_old
        # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
        # dot_product = dot_product - moving_dot_product

        hard_pseudo_label = torch.where(torch.sigmoid(t_logits_ul) > 0.5, 1.0, 0.0).reshape(-1,1)
        t_loss_mpl = dot_product * F.binary_cross_entropy_with_logits(t_logits_ul, hard_pseudo_label)
        t_loss = t_loss_mpl + t_loss_l

        t_loss.backward()
        t_optimizer.step()
        t_scheduler.step()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_mpl.update(t_loss_mpl.item())
        t_losses_l.update(t_loss_l.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. T_Loss_L : {t_losses_l.avg:.4f}. "
            f"T_Loss_MPL : {t_losses_mpl.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()

        if (step + 1) % 100 == 0:
            pbar.close()
            test_model = student_model
            y_true, y_pred = evaluate(args, test_model, test_loader, criterion_xent)
            accuracy, precision, recall, F1, AUROC, AUPRC, specificity= get_eval(y_true, y_pred)

            is_best = F1 >= best_f1
            if is_best:
                best_f1 = F1
                patience = 0
                best_wts = copy.deepcopy(student_model.state_dict())
                print("Update The Best Model")
                print(f"F1 Score : {best_f1:.2f}\n")
            else :
                patience += 1

            print(f'{step+1} Step')
            print(f"accuracy: {accuracy:.2f}, precision: {precision:.2f}")
            print(f"recall: {recall:.2f}, F1 score: {F1:.2f}")
            print(f"AUROC: {AUROC:.2f}, AUPRC: {AUPRC:.2f}")
            print(f"specificity: {specificity:.2f}\n")

            save_checkpoint(args, {
                'step': step + 1,
                'teacher_state_dict': teacher_model.state_dict(),
                'student_state_dict': student_model.state_dict(),
                'best_f1': best_f1,
                'teacher_optimizer': t_optimizer.state_dict(),
                'student_optimizer': s_optimizer.state_dict(),
                'teacher_scheduler': t_scheduler.state_dict(),
                'student_scheduler': s_scheduler.state_dict(),
            }, is_best)
            
            test_model = teacher_model
            y_true, y_pred = evaluate(args, test_model, test_loader, criterion_xent)
            
            if patience > early_stopping :
                break

    # finetune
    del t_scheduler, t_optimizer, teacher_model, labeled_loader, unlabeled_loader
    del s_scheduler, s_optimizer
    
    student_model.load_state_dict(best_wts)
    
    model = finetune(args, student_model, finetune_loader, test_loader, criterion_xent)
    
    return model
        
def predict(args, model, test_loader) :
    model.eval()
    test_iter = tqdm(test_loader)
    with torch.no_grad():
        y_pred = []
        y_true = []
        for step, (ecg, targets) in enumerate(test_iter):
            batch_size = ecg.shape[0]
            ecg = ecg.cuda()
            targets = targets.cuda()
        
            outputs = torch.sigmoid(model(ecg))
            
            y_pred.append(outputs.detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())
        test_iter.close()
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        
        return y_true, y_pred

def evaluate(args, model, test_loader, criterion_xent):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    F1 = AverageMeter()
    auroc = AverageMeter()
    specificity = AverageMeter()
    
    model.eval()
    test_iter = tqdm(test_loader)
    with torch.no_grad():
        end = time.time()
        y_pred = []
        y_true = []
        for step, (ecg, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = ecg.shape[0]
            ecg = ecg.cuda()
            targets = targets.cuda()
        
            outputs = model(ecg)
            loss = criterion_xent(outputs, targets)
            pred = torch.where(torch.sigmoid(outputs) > 0.5, 1.0, 0.0)
            acc, pre, rec, f1, AUROC, AUPRC, spec = get_eval(targets.detach().cpu().numpy(), pred.detach().cpu().numpy())
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            accuracy.update(acc, batch_size)
            precision.update(pre, batch_size)
            recall.update(rec, batch_size)
            F1.update(f1, batch_size)
            auroc.update(AUROC, batch_size)
            specificity.update(spec, batch_size)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"accuracy: {accuracy.avg:.2f}. precision: {precision.avg:.2f}. "
                f"recall: {recall.avg:.2f}. F1: {F1.avg:.2f}. "
                f"auroc: {auroc.avg:.2f}. specificity: {specificity.avg:.2f}. ")
            y_pred.append(pred.detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())
        test_iter.close()
        print(f"Test Loss : {losses.avg:.4f}")
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        
        return y_true, y_pred
    
def finetune(args, model, finetune_loader, test_loader, criterion_xent):
    optimizer_xent = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay,
                          nesterov=True)

    print("***** Running Finetuning *****")

    best_f1 = 0
    early_stopping = 30
    patience = 0
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(args.finetune_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(finetune_loader)
        for step, (ecg, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = ecg.shape[0]
            ecg = ecg.cuda()
            targets = targets.cuda()
            model.zero_grad()
            outputs = model(ecg)
            loss = criterion_xent(outputs, targets)

            loss.backward()
            optimizer_xent.step()

            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()

        y_true, y_pred = evaluate(args, model, test_loader, criterion_xent)
        accuracy, precision, recall, F1, AUROC, AUPRC, specificity= get_eval(y_true, y_pred)
        
        is_best = F1 >= best_f1
        if is_best:
            best_f1 = F1
            patience = 0
            best_wts = copy.deepcopy(model.state_dict())
            print("Update The Best Model")
            print(f"F1 Score : {best_f1:.2f}\n")
        else :
            patience += 1
        
        print(f"Finetune {epoch} Epoch")
        print(f"accuracy: {accuracy:.2f}, precision: {precision:.2f}")
        print(f"recall: {recall:.2f}, F1 score: {F1:.2f}")
        print(f"AUROC: {AUROC:.2f}, AUPRC: {AUPRC:.2f}")
        print(f"specificity: {specificity:.2f}\n\n")

        save_checkpoint(args, {
            'step': step + 1,
            'best_f1': best_f1,
            'student_state_dict': model.state_dict(),
            'student_optimizer': optimizer_xent.state_dict(),
        }, is_best, finetune=True)  
        
        if patience > early_stopping :
            break
        
    model.load_state_dict(best_wts)
    return model

def train_pl_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_loader,
               teacher_model, student_model, criterion_xent,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler) :

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # for author's code formula
    # moving_dot_product = torch.empty(1).cuda()
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    s_losses = AverageMeter()
    t_losses = AverageMeter()
    t_losses_l = AverageMeter()
    #t_losses_u = AverageMeter()
    t_losses_mpl = AverageMeter()
    mean_mask = AverageMeter()
    
    early_stopping = 20
    
    best_f1 = 0   
    patience = 0
    best_teacher_wts = copy.deepcopy(teacher_model.state_dict())

    for step in range(args.start_step, args.total_steps):
        if step % 100 == 0:
            pbar = tqdm(range(100))
        
        teacher_model.train()
        end = time.time()
        
        t_optimizer.zero_grad()

        try:
            ecg_l, targets = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            ecg_l, targets = next(labeled_iter)

        try:
            ecg_ul = next(unlabeled_iter)[0]
        except:
            unlabeled_iter = iter(unlabeled_loader)
            ecg_ul = next(unlabeled_iter)[0]

        data_time.update(time.time() - end)

        ecg_l = ecg_l.cuda()
        ecg_ul = ecg_ul.cuda()
        targets = targets.cuda()
     
        batch_size = ecg_l.shape[0]
        t_logits = teacher_model(ecg_l)

        t_loss_l = criterion_xent(t_logits, targets)

        t_loss = t_loss_l

        t_loss.backward()
        t_optimizer.step()
        t_scheduler.step()

        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. "
            f"T_Loss: {t_losses.avg:.4f}. "
            f"Mask: {mean_mask.avg:.4f}. ")
        pbar.update()

        if (step + 1) % 100 == 0:
            pbar.close()
            test_model = teacher_model
            y_true, y_pred = evaluate(args, test_model, test_loader, criterion_xent)
            accuracy, precision, recall, F1, AUROC, AUPRC, specificity= get_eval(y_true, y_pred)

            is_best = F1 > best_f1
            if is_best:
                best_f1 = F1
                patience = 0
                best_teacher_wts = copy.deepcopy(teacher_model.state_dict())
                print("Update The Best Model")
                print(f"F1 Score : {best_f1:.2f}\n")
            else :
                patience += 1
                
            if patience > early_stopping : 
                break

            print(f"accuracy: {accuracy:.2f}")
            print(f"precision: {precision:.2f}")
            print(f"recall: {recall:.2f}")
            print(f"F1 score: {F1:.2f}")
            print(f"AUROC: {AUROC:.2f}")
            print(f"AUPRC: {AUPRC:.2f}")
            print(f"specificity: {specificity:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'teacher_state_dict': teacher_model.state_dict(),
                'best_f1': best_f1,
                'teacher_optimizer': t_optimizer.state_dict(),
                'teacher_scheduler': t_scheduler.state_dict(),
            }, is_best)
    
    teacher_model.load_state_dict(best_teacher_wts)
    best_f1 = 0   
    patience = 0
    best_student_wts = copy.deepcopy(student_model.state_dict())
    
    for step in range(args.start_step, args.total_steps):
        if step % 100 == 0:
            pbar = tqdm(range(100))
        student_model.train()
        teacher_model.eval()
        end = time.time()
        s_optimizer.zero_grad()

        try:
            ecg_l, targets = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            ecg_l, targets = next(labeled_iter)

        try:
            ecg_ul = next(unlabeled_iter)[0]
        except:
            unlabeled_iter = iter(unlabeled_loader)
            ecg_ul = next(unlabeled_iter)[0]

        data_time.update(time.time() - end)

        ecg_l = ecg_l.cuda()
        ecg_ul = ecg_ul.cuda()
        targets = targets.cuda()
     
        batch_size = ecg_l.shape[0]
        t_ecg = torch.cat((ecg_l, ecg_ul))
        t_logits = teacher_model(t_ecg)
        t_logits_l = t_logits[:batch_size]
        t_logits_ul = t_logits[batch_size:]
        del t_logits

        t_loss_l = criterion_xent(t_logits_l, targets)

        soft_pseudo_label = torch.sigmoid(t_logits_ul.detach() / args.temperature)
        hard_pseudo_label = torch.where(soft_pseudo_label > 0.5, 1.0, 0.0).reshape(-1,1)
        mask = soft_pseudo_label.ge(0.5).float()

        s_ecg = torch.cat((ecg_l, ecg_ul))
        s_logits = student_model(s_ecg)
        s_logits_l = s_logits[:batch_size]
        s_logits_ul = s_logits[batch_size:]
        del s_logits

        s_loss_l_old = F.binary_cross_entropy_with_logits(s_logits_l, targets)
        
        s_loss = criterion_xent(s_logits_ul, hard_pseudo_label)
        
        s_loss = s_loss + s_loss_l_old
        
        s_loss.backward()
        
        s_optimizer.step()
        s_scheduler.step()
        
        s_losses.update(s_loss.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"Mask: {mean_mask.avg:.4f}. ")
        pbar.update()

        if (step + 1) % 100 == 0:
            pbar.close()
            test_model = student_model
            y_true, y_pred = evaluate(args, test_model, test_loader, criterion_xent)
            accuracy, precision, recall, F1, AUROC, AUPRC, specificity= get_eval(y_true, y_pred)

            is_best = F1 >= best_f1
            if is_best:
                best_f1 = F1
                patience = 0
                best_student_wts = copy.deepcopy(test_model.state_dict())
                
            else :
                patience += 1
                
            if patience > early_stopping : 
                break

            print(f"accuracy: {accuracy:.2f}")
            print(f"precision: {precision:.2f}")
            print(f"recall: {recall:.2f}")
            print(f"F1 score: {F1:.2f}")
            print(f"AUROC: {AUROC:.2f}")
            print(f"AUPRC: {AUPRC:.2f}")
            print(f"specificity: {specificity:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'teacher_state_dict': teacher_model.state_dict(),
                'student_state_dict': student_model.state_dict(),
                'best_f1': best_f1,
                'teacher_optimizer': t_optimizer.state_dict(),
                'student_optimizer': s_optimizer.state_dict(),
                'teacher_scheduler': t_scheduler.state_dict(),
                'student_scheduler': s_scheduler.state_dict(),
            }, is_best)

    # finetune
    del t_scheduler, t_optimizer, teacher_model, labeled_loader, unlabeled_loader
    del s_scheduler, s_optimizer
    
    student_model.load_state_dict(best_student_wts)
    
    model = finetune(args, student_model, finetune_loader, test_loader, criterion_xent)
    return model

def train_loop(model, criterion_xent, optimizer_xent, dataloaders, earlystopping=20, num_epochs=100, threshold=0.5):
    since = time.time()
    torch.autograd.set_detect_anomaly(True)

    dataset_sizes = {'train':len(dataloaders['train'].dataset),
                'val':len(dataloaders['val'].dataset)}

    best_model_wts = copy.deepcopy(model.state_dict()) # 베스트 모델의 가중치 복사
    best_acc = 0.0 # 베스트 정확도
    best_recall = 0.0
    best_loss = 1e8
    loss_list = {'train':[],
            'val':[]}
    patience = 0
    train_epochs = 0
    
    for epoch in tqdm(range(num_epochs)):
        train_epochs = epoch + 1
        print(f'Epoch {epoch+1}/{num_epochs}\n')
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval() # 모델을 평가 모드로 설정

            true_y = []
            pred_y = []

            running_loss = 0.0
            running_corrects = 0

            # 미니 배치 학습 루프
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer_xent.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    preds = (outputs >= threshold).float()*1
                    loss = criterion_xent(outputs, labels)

                    true_y.append(labels.detach().cpu())
                    pred_y.append(preds.detach().cpu())

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer_xent.step()
        
                running_loss += loss.item() * inputs.size(0)

            true_y = torch.cat(true_y, dim=0).numpy()
            pred_y = torch.cat(pred_y, dim=0).numpy()

            accuracy, precision, recall, f1, auroc, auprc, specificity = get_eval(true_y, pred_y)
           
            epoch_loss = running_loss / dataset_sizes[phase]
            
            loss_list[phase].append(epoch_loss)
            
            if phase == "val" :
                print(f"{phase} Loss: {epoch_loss:.4f}\n{phase} accuracy : {accuracy:.4f}\n{phase} precision : {precision:.4f}\
                    {phase} recall : {recall:.4f}\n{phase} f1_score : {f1:.4f}\n{phase} auroc : {auroc:.4f}\
                    {phase} auprc : {auprc:.4f}\n {phase} specificity : {specificity:.4f}\n")
            else :
                print(f"{phase} Loss: {epoch_loss:.4f}\n{phase} accuracy : {accuracy:.4f}\n")
                
            # 모델을 깊은 복사(deep copy)함
            if (phase == 'val') & (recall > best_recall):
                best_recall = recall
                print(f"Updating Model ! Best Recall Score {best_recall}")
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if (phase == 'val') & (epoch_loss < best_loss) :
                best_loss = epoch_loss
                
        print("Confusion Matrix")
        print(confusion_matrix(true_y, pred_y))
        if earlystopping != 0 :
            if best_loss < epoch_loss :
                patience = patience + 1
            else :
                patience = 0

            if patience > earlystopping :
                break
        print()
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    print(f'Best val Recall: {best_recall:4f}\n')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

def train_center_loop(model, criterion_xent, criterion_cent, optimizer_xent, optimizer_cent, dataloaders, earlystopping=20, num_epochs=100, threshold=0.5, alpha=1.0):
    since = time.time()
    torch.autograd.set_detect_anomaly(True)

    dataset_sizes = {'train':len(dataloaders['train'].dataset),
                'val':len(dataloaders['val'].dataset)}

    best_model_wts = copy.deepcopy(model.state_dict()) # 베스트 모델의 가중치 복사
    best_acc = 0.0 # 베스트 정확도
    best_recall = 0.0
    best_loss = 1e8
    loss_list = {'train':[],
            'val':[]}
    patience = 0
    train_epochs = 0
    
    for epoch in tqdm(range(num_epochs)):
        train_epochs = epoch + 1
        print(f'Epoch {epoch+1}/{num_epochs}\n')
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval() # 모델을 평가 모드로 설정

            true_y = []
            pred_y = []
            feat = []
            label = []

            running_loss = 0.0
            running_xent_loss = 0.0
            running_cent_loss = 0.0

            # 미니 배치 학습 루프
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                one_hot_labels = F.one_hot(labels.squeeze().to(torch.long), num_classes=2)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, feats = model(inputs)
                    
                    preds = (outputs >= threshold).float()*1
                    loss_xent = criterion_xent(outputs, labels)
                    loss_cent = criterion_cent(feats, one_hot_labels)
                    
                    #loss = loss_xent + alpha*loss_cent
                    loss = loss_cent

                    true_y.append(labels.detach().cpu())
                    pred_y.append(preds.detach().cpu())
                    feat.append(feats.detach().cpu())
                    label.append(one_hot_labels.detach().cpu())

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        #optimizer_xent.zero_grad()
                        optimizer_cent.zero_grad()
                        loss.backward()
                        #optimizer_xent.step()
                        for p in criterion_cent.parameters() :
                            p.grad.data *= (1. / alpha)
                        optimizer_cent.step()
        
                running_loss += loss.item() * inputs.size(0)
                running_xent_loss += loss_xent.item() * inputs.size(0)
                running_cent_loss += loss_cent.item() * inputs.size(0)

            true_y = torch.cat(true_y, dim=0).numpy()
            pred_y = torch.cat(pred_y, dim=0).numpy()
            feat = torch.cat(feat, dim=0).numpy()
            label = torch.cat(label, dim=0).numpy()

            accuracy, precision, recall, f1, auroc, auprc, specificity = get_eval(true_y, pred_y)
           
            epoch_loss = running_xent_loss / dataset_sizes[phase]
            center_loss = running_cent_loss / dataset_sizes[phase]
            
            loss_list[phase].append(epoch_loss)
            
            if phase == "val" :
                print(f"{phase} Loss: {epoch_loss:.4f}\n{phase} Center Loss: {center_loss:.4f}\n{phase} accuracy : {accuracy:.4f}\n{phase} precision : {precision:.4f}\
                    {phase} recall : {recall:.4f}\n{phase} f1_score : {f1:.4f}\n{phase} auroc : {auroc:.4f}\
                    {phase} auprc : {auprc:.4f}\n {phase} specificity : {specificity:.4f}\n")
                plot_features(feat, label, 2, epoch=epoch, prefix='Validation')
            else :
                plot_features(feat, label, 2, epoch=epoch, prefix="Train")
                print(f"{phase} Loss: {epoch_loss:.4f}\n{phase} Center Loss: {center_loss:.4f}\n{phase} accuracy : {accuracy:.4f}\n")
                
            # 모델을 깊은 복사(deep copy)함
            if (phase == 'val') & (recall > best_recall):
                best_recall = recall
                print(f"Updating Model ! Best Recall Score {best_recall}")
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if (phase == 'val') & (epoch_loss < best_loss) :
                best_loss = epoch_loss
                
        print("Confusion Matrix")
        print(confusion_matrix(true_y, pred_y))
        if earlystopping != 0 :
            if best_loss < epoch_loss :
                patience = patience + 1
            else :
                patience = 0

            if patience > earlystopping :
                break
        print()
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    print(f'Best val Recall: {best_recall:4f}\n')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

    