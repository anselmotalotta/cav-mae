# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# not rely on supervised feature

import sys
import os
import datetime
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import AverageMeter
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import wandb


def initialize_model(audio_model, device, args):
    """
    Initializes the model for training, setting it to the appropriate device and wrapping with DataParallel if necessary.
    Also sets up the optimizer and learning rate scheduler.
    """

    weight_decay = 5e-7
    betas = (0.95, 0.999)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    trainable_parameters = [p for p in audio_model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in audio_model.parameters())
    trainable_params = sum(p.numel() for p in trainable_parameters)

    print(f'Total parameter number is: {total_params / 1e6:.3f} million')
    print(f'Total trainable parameter number is: {trainable_params / 1e6:.3f} million')

    optimizer = torch.optim.Adam(trainable_parameters, args.lr, weight_decay=weight_decay, betas=betas)

    if args.lr_adapt:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Using adaptive learning rate scheduler.')
    else:
        milestones = range(args.lrscheduler_start, 1000, args.lrscheduler_step)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lrscheduler_decay)
        print(f'The learning rate scheduler starts at {args.lrscheduler_start} epoch with decay rate of {args.lrscheduler_decay} every {args.lrscheduler_step} epochs')

    return audio_model, optimizer, scheduler


def create_metrics_loggers():
    """
    Create a dictionary of AverageMeter instances for tracking training metrics.
    """
    metrics_loggers = {
        'batch_time': AverageMeter(),
        'per_sample_time': AverageMeter(),
        'data_time': AverageMeter(),
        'loss_av_meter': AverageMeter(),
        'loss_a_meter': AverageMeter(),
        'loss_v_meter': AverageMeter(),
        'loss_c_meter': AverageMeter()
    }
    return metrics_loggers

def reset_metric_loggers(metrics_loggers):
    """
    Reset the values in the AverageMeter instances in the metrics_loggers dictionary.
    """
    for meter in metrics_loggers.values():
        meter.reset()


def update_training_metrics(metrics_loggers, loss, loss_mae_a, loss_mae_v, loss_c, batch_size, dnn_start_time):
    """
    Update training metrics in the metrics_loggers dictionary.
    """
    metrics_loggers['loss_av_meter'].update(loss.item(), batch_size)
    metrics_loggers['loss_a_meter'].update(loss_mae_a.item(), batch_size)
    metrics_loggers['loss_v_meter'].update(loss_mae_v.item(), batch_size)
    metrics_loggers['loss_c_meter'].update(loss_c.item(), batch_size)
    metrics_loggers['per_sample_time'].update((time.time() - dnn_start_time) / batch_size)


def log_training_step(metrics_loggers, args, epoch, step, total_steps, c_acc):
    """
    Log information about the training step.
    """

    metrics = {
        "train/epoch": epoch,
        "train/loss_av": metrics_loggers["loss_av_meter"].val,
        "train/loss_mae_a": metrics_loggers["loss_a_meter"].val,
        "train/loss_mae_v": metrics_loggers["loss_v_meter"].val,
        "train/loss_c": metrics_loggers["loss_c_meter"].val,
        "train/c_acc": c_acc,
        "train/per_sample_time": metrics_loggers["per_sample_time"].avg
    }

    wandb.log(metrics)

    print_step = step % args.n_print_steps == 0
    early_print_step = epoch == 0 and step % (args.n_print_steps / 10) == 0
    final_step = step == (total_steps - 1)
    if print_step or early_print_step or final_step:
        print(f'Epoch: [{epoch}][{step}/{total_steps}]\t'
              f'Per Sample Total Time {metrics_loggers["per_sample_time"].avg:.5f}\t'
              f'Train Total Loss {metrics_loggers["loss_av_meter"].val:.4f}\t'
              f'Train MAE Loss Audio {metrics_loggers["loss_a_meter"].val:.4f}\t'
              f'Train MAE Loss Visual {metrics_loggers["loss_v_meter"].val:.4f}\t'
              f'Train Contrastive Loss {metrics_loggers["loss_c_meter"].val:.4f}\t'
              f'Train Contrastive Acc {c_acc:.3f}', flush=True)


def check_divergence(loss_av_meter):
    """
    Check if training has diverged (if the average loss is NaN).
    """
    return np.isnan(loss_av_meter.avg)


def train_epoch(epoch, audio_model, train_loader, optimizer, scaler, device, args, metrics_loggers):
    """
    Trains the model for one epoch.
    """
    for step, (a_input, v_input, _) in enumerate(tqdm(train_loader, desc="Training Progress")):
        step_start_time = time.time()
        batch_size = a_input.size(0)
        a_input = a_input.to(device, non_blocking=True)
        v_input = v_input.to(device, non_blocking=True)

        # Data loading time
        metrics_loggers['data_time'].update(time.time() - step_start_time)
        metrics_loggers['per_sample_time'].update((time.time() - step_start_time) / batch_size)

        # Forward pass and loss computation
        with autocast():
            loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(
                a_input, v_input, args.masking_ratio, args.masking_ratio,
                mae_loss_weight=args.mae_loss_weight,
                contrast_loss_weight=args.contrast_loss_weight,
                mask_mode=args.mask_mode
            )
            # Manually averaging loss for DataParallel
            loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = (
                loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            )

        # Backward pass and optimization
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        update_training_metrics(metrics_loggers, loss, loss_mae_a, loss_mae_v, loss_c, batch_size, step_start_time)

        # Logging step
        log_training_step(metrics_loggers, args, epoch, step, len(train_loader), c_acc)

        if check_divergence(metrics_loggers['loss_av_meter']):
            print("Training diverged...")
            return False

    return True


def validate(device, audio_model, val_loader, metrics_loggers, args):
    audio_model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, _) in enumerate(tqdm(val_loader)):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(
                    a_input, v_input, args.masking_ratio, args.masking_ratio,
                    mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight,
                    mask_mode=args.mask_mode
                )
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            A_loss.append(loss.detach().item())
            A_loss_mae.append(loss_mae.detach().item())
            A_loss_mae_a.append(loss_mae_a.detach().item())
            A_loss_mae_v.append(loss_mae_v.detach().item())
            A_loss_c.append(loss_c.detach().item())
            A_c_acc.append(c_acc.detach().item())
            end = time.time()

        loss = torch.mean(torch.tensor(A_loss))
        loss_mae = torch.mean(torch.tensor(A_loss_mae))
        loss_mae_a = torch.mean(torch.tensor(A_loss_mae_a))
        loss_mae_v = torch.mean(torch.tensor(A_loss_mae_v))
        loss_c = torch.mean(torch.tensor(A_loss_c))
        c_acc = torch.mean(torch.tensor(A_c_acc))

    eval_loss_av = loss.item()
    eval_loss_mae = loss_mae.item()
    eval_loss_mae_a = loss_mae_a.item()
    eval_loss_mae_v = loss_mae_v.item()
    eval_loss_c = loss_c.item()
    eval_c_acc = c_acc.item()

    val_metrics = {
            "val/eval_loss_av": eval_loss_av,
            "val/eval_loss_mae": eval_loss_mae,
            "val/eval_loss_mae_a": eval_loss_mae_a,
            "val/eval_loss_mae_v": eval_loss_mae_v,
            "val/eval_loss_c": eval_loss_c,
            "val/eval_c_acc": eval_c_acc,
    }
        
    wandb.log(val_metrics)

    print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
    print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
    print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
    print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
    print("Eval Total Loss: {:.6f}".format(eval_loss_av))
    print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

    print(f'Train Audio MAE Loss: {metrics_loggers["loss_a_meter"].avg:.6f}')
    print(f'Train Visual MAE Loss: {metrics_loggers["loss_v_meter"].avg:.6f}')
    print(f'Train Contrastive Loss: {metrics_loggers["loss_c_meter"].avg:.6f}')
    print(f'Train Total Loss: {metrics_loggers["loss_av_meter"].avg:.6f}')

    return eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    metrics_loggers = create_metrics_loggers()

    progress = []

    best_epoch, best_loss = 0, np.inf
    epoch = 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, best_epoch, best_loss,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    audio_model, optimizer, scheduler = initialize_model(audio_model, device, args)    

    print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

    epoch += 1
    scaler = GradScaler()

    print("start training...")
    result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print(f"current #epochs={epoch}")
        print('current masking ratio is {:.3f} for both modalities; audio mask mode {:s}'.format(args.masking_ratio, args.mask_mode))

        train_epoch_result = train_epoch(epoch, audio_model, train_loader, optimizer, scaler, device, args, metrics_loggers)

        if not train_epoch_result:
            return
        
        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(device, audio_model, test_loader, metrics_loggers, args)

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch-1, :] = [metrics_loggers["loss_a_meter"].avg, metrics_loggers["loss_v_meter"].avg, metrics_loggers["loss_c_meter"].avg, metrics_loggers["loss_av_meter"].avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        reset_metric_loggers(metrics_loggers)
