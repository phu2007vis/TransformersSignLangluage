import os
from pathlib import Path
import shutil
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

def copy_video(args):
    """Function to copy a single video file to its destination."""
    video_name, sub_folder, save_folder, phase = args
    try:
        _, cls_name, _ = video_name.split("_")
        cls_name = cls_name.split('P')[0].replace("A", "")
        sub_save_folder = os.path.join(save_folder, phase, cls_name)
        os.makedirs(sub_save_folder, exist_ok=True)
        
        old_video_path = os.path.join(sub_folder, video_name)
        new_video_path = os.path.join(sub_save_folder, video_name)
        shutil.copy(old_video_path, new_video_path)
        return True
    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        return False

def process_phase(phase):
    """Process all videos in a given phase."""
    sub_folder = os.path.join(root_folder, phase, "rgb")
    all_video_name = os.listdir(sub_folder)
    
    # Prepare arguments for multiprocessing
    args = [(video_name, sub_folder, save_folder, phase) for video_name in all_video_name]
    
    # Use multiprocessing Pool
    with Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(copy_video, args), total=len(args), desc=f"Processing {phase}"))
    
    return results

if __name__ == '__main__':
    root_folder = "/work/21013187/SAM-SLR-v2/data/rgb"
    save_folder = "/work/21013187/SAM-SLR-v2/data/phuoc_test_data"
    os.makedirs(save_folder, exist_ok=True)
    
    # Get all phases
    phases = os.listdir(root_folder)
    
    # Process each phase sequentially, but videos within each phase in parallel
    for phase in tqdm(phases, desc="Processing phases"):
        process_phase(phase)