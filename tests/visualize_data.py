import fire
from pathlib import Path

#add parrent dir to easy import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#model 
from dataset.landmark_transforms import Unnormalize
#dataloader 
from dataset import get_dataset
from dataset.utils import load_config
#metric funtion
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from dataset.video_transforms import VisUnnormalize
root_folder = os.path.dirname(os.path.dirname(__file__))
save_folder = os.path.join(root_folder,'visualize')

if os.path.exists(save_folder):
    import shutil
    shutil.rmtree(save_folder)
os.makedirs(save_folder)

def render_landmarks_to_image(landmarks, size=(224, 224)):
    """
    Plots (x, y) coordinates of landmarks and returns an RGB image.
    """
    fig, ax = plt.subplots()
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red')
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize(size)
    return np.array(img)

def reder_depth(depth_tensor, colormap=cv2.COLORMAP_JET):
    """
    Converts depth tensor to a colored video using a colormap for better visualization.
    Args:
        depth_tensor: torch.Tensor of shape (B, H, W)
        save_path: Output path (e.g., .mp4)
        fps: Frames per second
        colormap: OpenCV colormap (default: JET)
    """


    depth_tensor = depth_tensor.cpu().squeeze()
    B, H, W = depth_tensor.shape

    depth_np = depth_tensor.numpy()

    # Global min/max excluding zeros (invalid)
    valid_mask = depth_np > 0
    min_val = depth_np[valid_mask].min() if valid_mask.any() else 0
    max_val = depth_np.max()

    all_frames = []
    for i in range(B):
        frame = depth_np[i]

        # Normalize this frame (avoid invalid pixels)
        valid = frame > 0
        norm_frame = np.zeros_like(frame, dtype=np.float32)
        norm_frame[valid] = (frame[valid] - min_val) / (max_val - min_val + 1e-6)
        norm_frame = (norm_frame * 255).astype(np.uint8)

        # Apply colormap
        color_frame = cv2.applyColorMap(norm_frame, colormap)
        all_frames.append(color_frame)
    return all_frames
  

def save_landmarks_and_rgb_to_video(landmarks, rgb_array,depth, filename, fps=30):
    """
    landmarks_torch: torch.Tensor of shape (T, 27, 3)
    rgb_array: np.ndarray of shape (224, 224, 3, T)
    """
    T = landmarks.shape[0]
    # rgb_array = rgb_array.transpose(3, 0, 1, 2)  # (T, H, W, C)


    # Ensure RGB is uint8
    if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array ).clip(0, 255).astype(np.uint8)

    width, height = 224 * 3, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    all_depth_frames = reder_depth(depth)
    for t in range(T):
        rgb_frame = rgb_array[t]
        landmarks_frame = landmarks[t].numpy()
        vis_img = render_landmarks_to_image(landmarks_frame)
        depth_img = all_depth_frames[t]
        # Combine side-by-side
        combined = np.concatenate([vis_img, rgb_frame,depth_img], axis=1)  # (H, 2W, 3)
        # combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined)

    out.release()
    print(f"Saved video to {filename}")



def main(config_path = "/work/21013187/SAM-SLR-v2/phuoc_src/config/landmarks.yaml",
         num_visualize = 2):
    
    config = load_config(config_path)
    #dataset config
    dataset_config  = config['dataset']
    dataset_root_path = dataset_config['dataset_root_path']
    dataset_root_path = Path(dataset_root_path)
    
    #prepare model
    landmark_config = config['landmark']
    config['depth']['use'] = True

    landmark_config['use'] = True
    config['landmark'] = landmark_config
    landmark_unnormalize = Unnormalize()
    video_unnormalize = VisUnnormalize()
    train_dataset, _, _ = get_dataset(dataset_root_path, config=config)
    train_dataset = iter(train_dataset)
    for i in tqdm(range(num_visualize),total=num_visualize):
        video_name = f"{i}.mp4"
        save_path = os.path.join(save_folder,video_name)
        datadict = next(train_dataset)
        #3,T,224,224
        video_tensor = datadict['video']
        video_tensor = video_unnormalize(video_tensor)
        video = video_tensor.permute(1,2,3,0).numpy()
        # save_numpy_as_mp4(video,save_path)
        
        
        #T,27,3
        landmark = datadict['landmark']
        landmark = landmark_unnormalize(landmark)
        
        depth = datadict['depth']

        save_landmarks_and_rgb_to_video(landmark,video,depth ,save_path)
            
            
            
    
if __name__ == '__main__':
    fire.Fire(main)