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

def save_landmarks_and_rgb_to_video(landmarks, rgb_array, filename, fps=30):
    """
    landmarks_torch: torch.Tensor of shape (T, 27, 3)
    rgb_array: np.ndarray of shape (224, 224, 3, T)
    """
    T = landmarks.shape[0]
    # rgb_array = rgb_array.transpose(3, 0, 1, 2)  # (T, H, W, C)


    # Ensure RGB is uint8
    if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array ).clip(0, 255).astype(np.uint8)

    width, height = 224 * 2, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for t in range(T):
        rgb_frame = rgb_array[t]
        landmarks_frame = landmarks[t].numpy()
        vis_img = render_landmarks_to_image(landmarks_frame)

        # Combine side-by-side
        combined = np.concatenate([vis_img, rgb_frame], axis=1)  # (H, 2W, 3)
        # combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined)

    out.release()
    print(f"Saved video to {filename}")
    
def save_numpy_as_mp4(array, filename, fps=30):
    """
    Save a NumPy array of shape (224, 224, 3, T) as an MP4 video.

    Parameters:
        array (np.ndarray): NumPy array with shape (224, 224, 3, T)
        filename (str): Output filename, e.g. 'output.mp4'
        fps (int): Frames per second
    """
  
    T = array.shape[0]
    height, width = 224, 224

    # Convert from float to uint8 if necessary
    if array.dtype != np.uint8:
        array = (array ).clip(0, 255).astype(np.uint8)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for t in range(T):
        frame = array[t, :, :, :]
       
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        out.write(frame)

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
        
        save_landmarks_and_rgb_to_video(landmark,video,save_path)
            
            
            
    
if __name__ == '__main__':
    fire.Fire(main)