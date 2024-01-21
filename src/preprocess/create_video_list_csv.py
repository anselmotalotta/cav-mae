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
