# -*- coding: utf-8 -*-
# @Original Author : Yuan Gong
# @Affiliation     : Massachusetts Institute of Technology
# @Email           : yuangong@mit.edu
# @File            : dataloader.py
# @Time            : Original creation: 6/19/21 12:23 AM by Yuan Gong

# Modified and Further Developed by:
# @Author          : Anselmo Talotta
# @Email           : anselmo.talotta@gmail.com
#
# This file is based on open-source work originally created by Yuan Gong (https://github.com/YuanGongND/cav-mae).
# Modifications and additional features have been implemented by Anselmo Talotta.
#
# Portions of this code are modified from:
# Author: David Harwath
# Some functions are borrowed from https://github.com/SeanNaren/deepspeech.pytorch


import csv
import json
import os.path
import glob

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL

def make_index_dict(label_csv):
    """
    Creates a dictionary that maps label mid to an index.

    Parameters:
    - label_csv (str): Path to the CSV file containing label information with 'mid' and 'index' columns.

    Returns:
    - index_lookup (dict): A dictionary where each key is a 'mid' from the CSV file, and its value is the corresponding 'index'.
    """
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    """
    Creates a dictionary that maps an index to a display name.

    Parameters:
    - label_csv (str): Path to the CSV file containing label information with 'index' and 'display_name' columns.

    Returns:
    - name_lookup (dict): A dictionary where each key is an 'index' from the CSV file, and its value is the corresponding 'display_name'.
    """
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    """
    Converts a list of indices to their corresponding display names using a label CSV file.

    Parameters:
    - index_list (list of int): A list of indices.
    - label_csv (str): Path to the CSV file containing label information with 'index' and 'display_name' columns.

    Returns:
    - label_list (list of str): A list of display names corresponding to the indices in the index_list.
    """
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """
    Applies a preemphasis filter to the input signal. This filter increases the amplitude of high-frequency bands and decreases the amplitude of lower bands.

    Parameters:
    - signal (numpy.ndarray): The input signal to filter.
    - coeff (float): The preemphasis coefficient. 0 is no filter, default is 0.97.

    Returns:
    - numpy.ndarray: The filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Initializes the AudiosetDataset object with the dataset, audio configurations, and label information.

        Parameters:
        - dataset_json_file (str): Path to the JSON file containing the dataset. The file should list audio recordings and their metadata.
        - audio_conf (dict): A dictionary containing various audio configuration settings such as sample rate, Mel bin count, frequency masking parameters, 
        mixup rate, normalization stats, noise augmentation flag, and more.
        - label_csv (str, optional): Path to a CSV file containing label information. Used for creating label index and name dictionaries.

        The initialization process involves:
        - Loading and preprocessing the dataset.
        - Extracting and setting up audio configuration settings.
        - Preparing image preprocessing pipeline.
        - Initializing parameters for audio processing, data augmentation, and normalization.
        - Setting up label index mapping and determining the number of classes.
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get('total_frame', 10)
        print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        """
        Converts a list of data entries into a NumPy array. This is done to manage memory more efficiently.

        Parameters:
        - data_json (list): A list of data entries, where each entry is a dictionary containing keys like 'wav', 'labels', 'video_id', and 'video_path'.

        Returns:
        - data_np (numpy.ndarray): A NumPy array of the same data, where each entry is a list of the values from the data dictionaries.
        """
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        """
        Converts a waveform file to filter bank features. Supports mixup by combining two waveforms.

        Parameters:
        - filename (str): Path to the main waveform file.
        - filename2 (str, optional): Path to the second waveform file for mixup. Defaults to None.
        - mix_lambda (float, optional): The mixup lambda for weighting the combination of two waveforms. Defaults to -1 (no mixup).

        Returns:
        - fbank (torch.Tensor): The filter bank features extracted from the waveform.
        """
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank
    

    def randselect_img(self, video_id, video_path):
        """
        Selects a frame image from a video. In 'eval' mode, it selects a specified frame or the middle frame by default. In training mode, it selects a random frame.

        Parameters:
        - video_id (str): The ID of the video.
        - video_path (str): The file path where video frames are stored.

        Returns:
        - out_path (str): The file path of the selected frame image.
        """

        def build_frame_path(video_path, frame_idx, video_id):
            return  os.path.join(video_path, 'frame_' + str(frame_idx), video_id + '*.jpg')


        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 9)

        frame_path = build_frame_path(video_path, frame_idx, video_id)
        possible_files = glob.glob(frame_path)

        while not possible_files and frame_idx >= 0:
            # print('frame {:s} {:d} does not exist'.format(video_id, frame_idx))
            frame_idx -= 1
            frame_path = build_frame_path(video_path, frame_idx, video_id)
            possible_files = glob.glob(frame_path)

        # If still no files found, return None or a default image path
        if not possible_files:
            print(f'No image file found for video_id {video_id}.')
            return None  # Or return a path to a default image

        out_path = possible_files[0]
        # print(out_path)
        return out_path
    
    def process_audio_image(self, datum, mix_datum=None, mix_lambda=0):
        """Processes audio and image data, handling mixup if necessary."""
        try:
            fbank = self._wav2fbank(datum['wav'], mix_datum['wav'] if mix_datum else None, mix_lambda)
        except Exception as e:
            fbank = torch.zeros([self.target_length, 128]) + 0.01
            # print('there is an error in loading audio', e)

        try:
            image = self.get_image(
                self.randselect_img(datum['video_id'], datum['video_path']),
                self.randselect_img(mix_datum['video_id'], mix_datum['video_path']) if mix_datum else None,
                mix_lambda
            )
        except Exception as e:
            image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
            # print('there is an error in loading image', e)

        return fbank, image
    
    def process_labels(self, labels_string, mix_labels_string=None, mix_lambda=1):
        """
        Processes the label string into a tensor of label indices.
        Handles both single-label processing and mixup label processing.
        """
        # print(f"labels received: {labels_string}, mix_labels received: {mix_labels_string}, mix_lambda: {mix_lambda}")
        label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)

        confidence_weight = 1.0 - self.label_smooth
        
        for label_str in labels_string.split(','):
            label_str = label_str.strip().replace('"', '')
            if label_str in self.index_dict:
                label_index = int(self.index_dict[label_str])
                label_indices[label_index] += mix_lambda * confidence_weight 

        # Handling mixup case
        if mix_labels_string is not None:
            for mix_label_str in mix_labels_string.split(','):
                mix_label_str = label_str.strip().replace('"', '')
                mix_label_str = mix_label_str.strip()
                if mix_label_str in self.index_dict:
                    mix_label_index = int(self.index_dict[mix_label_str])
                    label_indices[mix_label_index] += (1.0 - mix_lambda) * confidence_weight

        label_indices = torch.FloatTensor(label_indices)
        return label_indices
    
    def apply_spec_augment(self, fbank):
        """
        Applies SpecAugment to the filter bank features.
        Frequency and time masking are applied if their respective parameters are non-zero.
        This augmentation should not be applied in evaluation mode.

        Parameters:
        - fbank (torch.Tensor): The filter bank features to augment.

        Returns:
        - torch.Tensor: The augmented filter bank features.
        """
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        return fbank
    
    def apply_noise_augmentation(self, fbank):
        """
        Applies noise augmentation to the filter bank features. This augmentation adds random noise and 
        performs a random roll (shift) on the features. It's typically used during training to improve 
        the robustness of the model.

        Parameters:
        - fbank (torch.Tensor): The filter bank features to augment.

        Returns:
        - torch.Tensor: The noise-augmented filter bank features.
        """
        # Add random noise
        noise_level = np.random.rand() / 10
        fbank += torch.rand(fbank.shape[0], fbank.shape[1]) * noise_level

        # Randomly roll the features
        fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        return fbank
    
    def get_decoded_datum(self, index):
        """
        Retrieves and decodes a single data point from the dataset at the specified index.

        Parameters:
        - index (int): The index of the data point to be retrieved.

        Returns:
        - dict: The decoded data point, typically in a dictionary format containing relevant information such as audio paths, labels, etc.
        """
        raw_datum = self.data[index]
        decoded_datum = self.decode_data(raw_datum)
        return decoded_datum


    def __getitem__(self, index):
        """
        Retrieves a single data point from the dataset based on the given index. This involves processing both the audio and image data, applying various 
        augmentations, and preparing the label.

        Parameters:
        - index (int): The index of the data point to be retrieved.

        Returns:
        - (torch.Tensor, torch.Tensor, torch.FloatTensor): A tuple containing the processed audio features (filter bank), the processed image, and the label 
        indices in one-hot encoded format. The audio features and images undergo various augmentations and preprocessing steps based on the configuration settings.

        The process includes:
        - Handling mixup augmentation for both audio and image data if enabled.
        - Applying label smoothing.
        - Conducting SpecAugment on the audio features.
        - Applying noise augmentation if enabled.
        - Normalizing the filter bank features.
        - Handling exceptions in audio and image loading with fallbacks.
        """
        datum = self.get_decoded_datum(index)
        if random.random() < self.mixup and self.mode != 'eval':
            mix_datum = self.get_decoded_datum(random.randint(0, self.num_samples-1))
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            fbank, image = self.process_audio_image(datum, mix_datum, mix_lambda)
            label_indices = self.process_labels(datum['labels'], mix_datum['labels'], mix_lambda)
        else:         
            fbank, image = self.process_audio_image(datum)
            label_indices = self.process_labels(datum['labels'])

        # SpecAug, not do for eval set
        if self.mode != 'eval':
            fbank = self.apply_spec_augment(fbank)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std)

        # Noise augmentation
        if self.noise:
            fbank = self.apply_noise_augmentation(fbank)

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, image, label_indices

    def __len__(self):
        return self.num_samples