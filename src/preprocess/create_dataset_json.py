import argparse
import json
import os
import csv
import pytrie


directory_cache = {}

def find_file_with_prefix(folder_path, file_prefix):
    if folder_path not in directory_cache:
        trie = pytrie.SortedStringTrie()
        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            trie[file_name] = full_path
        directory_cache[folder_path] = trie

    # Finding all items in the trie that start with the given prefix
    matches = directory_cache[folder_path].items(prefix=file_prefix)
    # Return the full path of the first match, if any
    file_path =  next(iter(matches))[1] if matches else None
    return file_path


def generate_json(labels_csv, video_folder, audio_folder, dataset_json_file):
    new_data = []

    # Reading CSV
    with open(labels_csv, 'r', encoding='utf-8') as file:
        # Use a generator expression to skip lines starting with '#'
        filtered_lines = (line for line in file if not line.startswith('#'))
        
        # Create a CSV reader with the filtered lines
        csv_reader = csv.DictReader(filtered_lines)
        for row in csv_reader:
            video_id = row['YTID']
            video_path = video_folder
            wav = find_file_with_prefix(audio_folder, video_id)
            labels = row['positive_labels']

            if wav is not None:
                new_entry = {}
                new_entry['video_id'] = video_id
                new_entry['wav'] = wav
                new_entry['video_path'] = video_path
                new_entry['labels'] = labels
                new_data.append(new_entry)

    output = {'data': new_data}

    print(f'Creating output JSON file after clean {len(new_data)} files')
    with open(dataset_json_file, 'w') as f:
        json.dump(output, f, indent=1)

def main():
    parser = argparse.ArgumentParser(description="Generate a JSON audio / video dataset companion that will facilitate the load of the data during training or validation.")
    parser.add_argument("--label_csv_file", help="Path to file containing the labels of the audio / video dataset.")
    parser.add_argument("--video_folder", help="Path to folder containing the videos of the dataset.")
    parser.add_argument("--audio_folder", help="Path to folder containing the audios of the dataset.")
    parser.add_argument("--output", default="file_paths.json", help="Name of the output JSON file (default: file_paths.json)")

    args = parser.parse_args()

    # Resolving the absolute path if a relative path is given
    abs_video_folder = os.path.abspath(args.video_folder)
    abs_audio_folder = os.path.abspath(args.audio_folder)

    print("abs_video_folder", abs_video_folder)
    print("abs_audio_folder", abs_audio_folder)

    generate_json(args.label_csv_file, abs_video_folder, abs_audio_folder, args.output)

#esample: 
# python create_dataset_json.py --label_csv_file /storage/data/cavmae/audioset/balanced_train_segment_labels.csv --video_folder /storage/data/cavmae/audioset/preprocesses/balanced_train_segments/video_frames --audio_folder /storage/data/cavmae/audioset/preprocesses/balanced_train_segments/sample_audio --output /storage/data/cavmae/audioset/audioset_20k_custom.json

# python create_dataset_json.py --label_csv_file /storage/data/cavmae/audioset/eval_segment_labels.csv --video_folder /storage/data/cavmae/audioset/preprocesses/eval_segments_partial/video_frames --audio_folder /storage/data/cavmae/audioset/preprocesses/eval_segments_partial/sample_audio --output /storage/data/cavmae/audioset/audioset_eval_partial_custom.json

if __name__ == "__main__":
    main()
