import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Easy video feature extractor')
parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="Should be a csv file of a single columns, each row is the input video path.")
parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="The place to store the video frames.")
args = parser.parse_args()

input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)
if os.path.exists(args.target_fold) == False:
    os.makedirs(args.target_fold)

for i in range(input_filelist.shape[0]):
    input_f = input_filelist[i]
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    video_id = input_f.split('/')[-1][:-ext_len-1]
    output_f_1 = args.target_fold + '/' + video_id + '_intermediate.wav'
    output_f_2 = args.target_fold + '/' + video_id + '.wav'
    if os.path.exists(output_f_2):
        print(f"File {output_f_2} already exists, skipping")
        continue
    # first resample audio
    if os.path.exists(output_f_1):
        print(f"File {output_f_1} already exists, skipping")
    else:
        print(f"File {output_f_1} doesn't exists, creating")
        os.system('ffmpeg -i {:s} -vn -ar 16000 {:s} > ffmpeg.log'.format(input_f, output_f_1)) # save an intermediate file

    # then extract the first channel
    if os.path.exists(output_f_1):
        print(f"Creating {output_f_2}")
        os.system('sox {:s} {:s} remix 1  > sox.log'.format(output_f_1, output_f_2))
        os.remove(output_f_1)
    else:
        print("WARNING: Failure in creating", output_f_1)
