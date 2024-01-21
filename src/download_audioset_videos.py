# From: https://gist.github.com/wsntxxn/6a79c3a73b55fe11e55720032fa0b266

import argparse
import os
import time
from pathlib import Path

import yt_dlp
from yt_dlp import DownloadError
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError

unavail_msgs = [
    "The uploader has not made this video available in your country.",
    "This video requires payment to watch.",
    "Private video",
    "Video unavailable",
    "This video has been removed"
]

@timeout_decorator.timeout(180)
def download_video(yid, download_path, proxy, unavail_path, unavail_list):
    if yid in unavail_list:
        print(f"{yid} unavailable on youtube")
        return
    opts = {
        # "format": "bestvideo[ext=mp4][height<=480]/best[ext=mp4][height<=480]/best",
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(download_path, "%(id)s.%(ext)s"),
        "proxy": proxy
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download(f"https://youtube.com/watch?v={yid}")
    except DownloadError as e:
        unavailable = False
        for msg in unavail_msgs:
            if msg in e.msg:
                unavailable = True
                break
        if unavailable:
            with open(unavail_path, "a") as writer:
                writer.write(yid + "\n")
    time.sleep(10)


def trim_video(download_path, yid, start, end):
    downloaded_file = f"{download_path}/{yid}.mp4"
    outname = os.path.join(download_path, f"{yid}_{start}_{end}.mp4")
    command = f"ffmpeg -y -loglevel quiet -i {downloaded_file} -ss {start} -to {end} {outname}"
    os.system(command)

def trim_and_convert_video(download_path, yid, start, end):
    downloaded_file_mp4 = f"{download_path}/{yid}.mp4"
    downloaded_file_webm = f"{download_path}/{yid}.webm"
    downloaded_file_mkv = f"{download_path}/{yid}.mkv"
    outname = os.path.join(download_path, f"{yid}_{start}_{end}.mp4")

    # Determine if the file is in webm or mp4 format
    if os.path.exists(downloaded_file_webm):
        downloaded_file = downloaded_file_webm
        command = f"ffmpeg -y -loglevel quiet -i {downloaded_file} -ss {start} -to {end} -c:v libx264 -c:a aac {outname}"
    elif os.path.exists(downloaded_file_mkv):
        downloaded_file = downloaded_file_mkv
        command = f"ffmpeg -y -loglevel quiet -i {downloaded_file} -ss {start} -to {end} -c:v libx264 -c:a aac {outname}"
    elif os.path.exists(downloaded_file_mp4):
        downloaded_file = downloaded_file_mp4
        command = f"ffmpeg -y -loglevel quiet -i {downloaded_file} -ss {start} -to {end} {outname}"
    else:
        print(f"No downloaded file found for {yid}")
        return

    # Use ffmpeg to trim and convert (if necessary) to mp4
    os.system(command)

    # Remove the original downloaded file
    os.remove(downloaded_file)


def download_videos(args):
    download_path = args.download_path
    proxy = args.proxy
    input_csv = args.input_csv
    stem = Path(input_csv).stem
    unavail_path = Path(input_csv).with_name(f"{stem}_unavail.csv")
    if not os.path.exists(unavail_path):
        writer = open(unavail_path, "w")
        writer.close()
    with open(unavail_path, "r") as reader:
        unavail_list = [line.strip() for line in reader.readlines()]
    if not os.path.exists(download_path):
        os.mkdir(download_path)
    with open(input_csv, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            if line.startswith("# "):
                continue
            yid, start, end, _ = line.strip().split(", ")
            outname = os.path.join(download_path, f"{yid}_{start}_{end}.mp4")
            if os.path.exists(outname):
                print(f"Found file {outname}")
            else:
                try:
                    download_video(yid, download_path, proxy, unavail_path, unavail_list)
                except TimeoutError:
                    pass

            trim_and_convert_video(download_path, yid, start, end)

# launch with nohup /home/anselmo/miniconda3/envs/cavmae/bin/python /home/anselmo/workspace/cav-mae/src/download_audioset_videos.py --input_csv  /storage/data/cavmae/audioset/balanced_train_segments.csv --download_path /storage/data/cavmae/audioset/videos/balanced_train_segments --proxy ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str)
    parser.add_argument("--download_path", type=str)
    parser.add_argument("--proxy", type=str)

    args = parser.parse_args()
    download_videos(args)