import os
import csv
import argparse

def generate_file_list_csv(folder, csv_filename):
    # Making the folder path relative to the script location

    # Collecting file paths
    file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # Writing to CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for path in file_paths:
            csvwriter.writerow([path])

def main():
    parser = argparse.ArgumentParser(description="Generate a CSV file of file paths in a folder.")
    parser.add_argument("--folder", help="Folder to list files from, relative to this script.")
    parser.add_argument("--output", default="file_paths.csv", help="Name of the output CSV file (default: file_paths.csv)")

    args = parser.parse_args()

    # Resolving the absolute path if a relative path is given
    abs_folder = os.path.abspath(args.folder)

    print("abs_folder", abs_folder)

    generate_file_list_csv(abs_folder, args.output)

if __name__ == "__main__":
    main()

"""

cd cav-mae/src/preprocess

python create_video_list_csv.py --folder /storage/data/cavmae/audioset/videos/eval_segments_partial --output /storage/data/cavmae/eval_segments_partial_list.csv

# extract video frames
python extract_video_frame.py -input_file_list /storage/data/cavmae/audioset/eval_segments_partial_list.csv -target_fold /storage/data/cavmae/audioset/preprocesses/eval_segments_partial/video_frames
# extract audio tracks
python extract_audio.py  -input_file_list /storage/data/cavmae/audioset/eval_segments_partial_list.csv -target_fold /storage/data/cavmae/audioset/preprocesses/eval_segments_partial/sample_audio

"""