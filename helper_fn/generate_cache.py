import os
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from tqdm import tqdm

def video_loader(video_path):
    vidcap = cv2.VideoCapture(str(video_path))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)

    idx = 0
    success, image = vidcap.read()
    while success and idx < frame_count:
        frames[idx] = image
        success, image = vidcap.read()
        idx += 1

    vidcap.release()

    frames = frames[:idx]
    return torch.from_numpy(frames).permute(3, 0, 1, 2).float()

def save_video_as_npy(video_tensor, video_path):
    video_array = video_tensor.permute(1, 2, 3, 0).to(torch.uint8).numpy()
    np.save(video_path, video_array)

def process(video_path):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "cache")
    id = os.path.splitext(os.path.basename(video_path))[0]
    base_cache_name = id + ".npy"
    cache_video_path = os.path.join(cache_dir, base_cache_name)

    video = video_loader(video_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)  # Safer for concurrent access
    save_video_as_npy(video, cache_video_path)
import os
import cv2
import numpy as np
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from tqdm import tqdm

def video_loader(video_path):
    vidcap = cv2.VideoCapture(str(video_path))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)

    idx = 0
    success, image = vidcap.read()
    while success and idx < frame_count:
        frames[idx] = image
        success, image = vidcap.read()
        idx += 1

    vidcap.release()

    frames = frames[:idx]
    return torch.from_numpy(frames).permute(3, 0, 1, 2).float()

def save_video_as_npy(video_tensor, video_path):
    video_array = video_tensor.permute(1, 2, 3, 0).to(torch.uint8).numpy()
    np.save(video_path, video_array)

def process(video_path):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "cache")
    id = os.path.splitext(os.path.basename(video_path))[0]
    base_cache_name = id + ".npy"
    cache_video_path = os.path.join(cache_dir, base_cache_name)

    video = video_loader(video_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    save_video_as_npy(video, cache_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos in parallel.")
    parser.add_argument('--max_num_classes', type=int, default=300, help="Maximum number of classes")
    parser.add_argument('--root_folder', type=str, default="/work/21013187/SAM-SLR-v2/data/rgb", help="Root folder for video data")
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    all_class_index = range(args.max_num_classes)
    exts = ['.avi', '.mp4']

    for directory_name in os.listdir(args.root_folder):
        if directory_name =='test':
            continue
        directory = os.path.join(args.root_folder, directory_name, 'rgb')
        for class_index in tqdm(all_class_index, total=args.max_num_classes):
            data_target_path = []
            for ext in exts:
                data_target_path.extend(glob(os.path.join(directory, f"*A{class_index}P*{ext}")))
            print(f"Split: {directory_name} index: {class_index+1} has {len(data_target_path)} files!")
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_path = {executor.submit(process, path): path for path in sorted(data_target_path)}
                
                for future in tqdm(as_completed(future_to_path), total=len(data_target_path), desc="Processing videos"):
                    path = future_to_path[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing {path}: {e}")