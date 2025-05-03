import os
import torch

from dataset.video_transforms import (
    ApplyTransformToKey,
    Normalize as VideoNormalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    RemoveKey
)
from dataset.depth_transforms import DepthResize

from dataset.landmark_transforms import (
    Normalize as LandmarkNormalize,
    RandomHorizontalFlip as LandmarkRandomHorizontalFlip,
    UniformTemporalSubsample as LandmarkUniformTemporalSubsample,
    RandomCrop as LandmarkRandomCrop,
    Resize as LandmarkResize
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
)
from torchvision import transforms
import pytorchvideo
from dataset.utils import labeled_video_dataset

def get_dataset(dataset_root_path,
                img_size = (224, 224),
                mean = [0.5,0.5,0.5],
                std = [0.5,0.5,0.5],
                num_frames = 16,
                config = None):

    video_count_train = len(list(dataset_root_path.glob("train/rgb/*.avi")))
    video_count_val = len(list(dataset_root_path.glob("val/rgb/*.avi")))
    video_count_test = len(list(dataset_root_path.glob("test/rgb/*.avi")))
    video_total = video_count_train + video_count_val + video_count_test

    print(f"Total videos: {video_total}")
    
    # chua biet cai nay lam gi :)) nhung dung xoa
    sample_rate = 4
    fps = 30
    clip_duration = num_frames * sample_rate / fps

    # -- NOTE -- Tung: transfrom tren vid -> transform tren landmark tuong tu k (cung gia tri so voi vid)
    
    # Training dataset transformations.
    train_transform = Compose(
        [
           
            #remove unuse key
            RemoveKey(['video_index','clip_index','aug_index','video_name']),
             # apply all the key
            UniformTemporalSubsample(num_frames),
            # apply just only key focus
            # RandomHorizontalFlip(),
            ApplyTransformToKey(
                key = 'video',
                transform=Compose(
                    [
                        transforms.RandomHorizontalFlip(p =0),
                        Lambda(lambda x: x / 255.0),
                        VideoNormalize(mean, std),
                        Resize(img_size),
                    ]
                ),
            ),

            ApplyTransformToKey(
                key = 'landmark',
                transform=Compose(
                    [
                        LandmarkNormalize(mean=(0, 0, 0), std=(1, 1, 1)),  
                    ]
                ),
            ),
            ApplyTransformToKey(
                key = 'depth',
                transform=Compose(
                    [   
                        DepthResize(),
                        Lambda(lambda x: (x-x.min())/(x.max()-x.min())),
                      
                    ]
                ),
            ),
        ]
    )
    
    # Training dataset.
    train_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "train",'rgb'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
        config = config
    )

    # Validation and evaluation datasets' transformations.
    val_transform = Compose(
        [
            RemoveKey(['video_index','clip_index','aug_index','video_name']),
            UniformTemporalSubsample(num_frames),
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Lambda(lambda x: x / 255.0),
                        VideoNormalize(mean, std),
                        Resize(img_size),
                    ]
                ),
            ),
            # Add transform for landmark - validation set
            ApplyTransformToKey(
                key = 'landmark',
                transform=Compose(
                    [
                        LandmarkNormalize(mean=(0, 0, 0), std=(1, 1, 1)),
                    ]
                ),
            ),
             ApplyTransformToKey(
                key = 'depth',
                transform=Compose(
                    [   
                        DepthResize(),
                        Lambda(lambda x: (x-x.min())/(x.max()-x.min())),
                      
                    ]
                ),
            ),
        ]
    )

    # Validation and evaluation datasets.
    val_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "val",'rgb'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
        config =config
    )

    test_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "test",'rgb'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
        config =config
    )
    
    return train_dataset,val_dataset,test_dataset




def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values =[example["video"].permute(1, 0, 2, 3) for example in examples ]
    pixel_values =  torch.stack(pixel_values) 
    
    labels = torch.tensor([example["label"] for example in examples],dtype=torch.long)
    batch_dict =  {"pixel_values": pixel_values, "labels": labels}
    
    landmarks = [example["landmark"] for example in examples if example.get('landmark') is not None]
    landmarks = torch.stack(landmarks) if len(landmarks) else None
    if landmarks is not None:
        batch_dict['landmarks'] = landmarks
        
    depths = [example["depth"] for example in examples if example.get('depth') is not None]
    depths = torch.stack(depths) if len(depths) else None
    if depths is not None:
        batch_dict['depths'] = depths
        
    return batch_dict