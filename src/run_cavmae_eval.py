# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
import warnings
from tqdm import tqdm
from sklearn import metrics
from validate_model import validate

# finetune cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400"])
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
# not used in the formal experiments, only in preliminary experiments
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)

args = parser.parse_args()


def load_model(args):
    """Load the specified model."""
    if args.model == 'cav-mae-ft':
        print('evaluate a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
        model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
    else:
        raise ValueError('Model not supported.')

    if args.pretrain_path != 'None':
        model_weights = torch.load(args.pretrain_path)
        model = torch.nn.DataParallel(model)
        missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
        print(f'Loaded pretrained weights from {args.pretrain_path}, missing keys: {missing_keys}, unexpected keys: {unexpected_keys}')
    else:
        warnings.warn("Evaluating a model without any training.")
    return model


def create_data_loader(args, audio_conf):

    """Create DataLoader for the dataset."""

    loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    return loader


def evaluate_model(model, args, val_audio_conf, frame=None):
    """
    Evaluate the model for a given frame or single evaluation if frame is None.
    Args:
        model: The audio model to evaluate.
        args: Command line arguments passed to the script.
        frame (Optional[int]): The specific frame to evaluate on. If None, evaluates on single configuration.

    Returns:
        tuple: A tuple containing the current result and the audio output.
    """

    # Create DataLoader
    val_loader = create_data_loader(args, val_audio_conf)

    # Perform validation
    stats, audio_output, target = validate(model, val_loader, args)

    print(audio_output.shape)
    if args.metrics == 'acc':
        audio_output = torch.nn.functional.softmax(audio_output.float(), dim=-1)
    elif args.metrics == 'mAP':
        audio_output = torch.nn.functional.sigmoid(audio_output.float())

    # Process results
    if args.metrics == 'mAP':
        result = np.mean([stat['AP'] for stat in stats])
        print(f'mAP{" of frame " + str(frame) if frame is not None else ""} is {result:.4f}')
    elif args.metrics == 'acc':
        result = stats[0]['acc']
        print(f'acc{" of frame " + str(frame) if frame is not None else ""} is {result:.4f}')

    return result, audio_output.numpy(), target.numpy()



audio_model = load_model(args)

# all exp in this work is based on 224 * 224 image
im_res = 224
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

# skip multi-frame evaluation, for audio-only model
if args.skip_frame_agg:
    print('skip multi-frame evaluation')
    val_audio_conf['frame_use'] = 5
    evaluate_model(audio_model, args, val_audio_conf)
else:
    print('multi-frame evaluation')
    result = []
    multiframe_pred = []
    total_frames = 10 # change if your total frame is different
    for frame in tqdm(range(total_frames), desc="Multi-frame evaluation"):
        val_audio_conf['frame_use'] = frame
        current_result, audio_output, target = evaluate_model(audio_model, args, val_audio_conf, frame=frame)
        
        multiframe_pred.append(audio_output)
        
        result.append(current_result)

    # ensemble over frames
    multiframe_pred = np.mean(multiframe_pred, axis=0)
    if args.metrics == 'acc':
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(multiframe_pred, 1))
        print('multi-frame acc is {:f}'.format(acc))
        result.append(acc)
    elif args.metrics == 'mAP':
        AP = []
        for k in range(args.n_class):
            # Average precision
            avg_precision = metrics.average_precision_score(target[:, k], multiframe_pred[:, k], average=None)
            AP.append(avg_precision)
        mAP = np.mean(AP)
        print('multi-frame mAP is {:.4f}'.format(mAP))
        result.append(mAP)
    np.savetxt(args.exp_dir + '/mul_frame_res.csv', result, delimiter=',')