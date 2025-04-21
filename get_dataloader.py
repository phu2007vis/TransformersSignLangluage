import os
import torch

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

import pytorchvideo
from dataset_phuoc import labeled_video_dataset

def get_dataset(dataset_root_path,
                   model,
                   image_processor):

    video_count_train = len(list(dataset_root_path.glob("train/rgb/*.avi")))
    video_count_val = len(list(dataset_root_path.glob("val/rgb/*.avi")))
    video_count_test = len(list(dataset_root_path.glob("test/rgb/*.avi")))
    video_total = video_count_train + video_count_val + video_count_test

    print(f"Total videos: {video_total}")



    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
        
    resize_to = (height, width)
    
    mean = image_processor.image_mean
    std = image_processor.image_std
    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps


    # Training dataset transformations.
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
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
    )

    # Validation and evaluation datasets' transformations.
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
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
    )

    test_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "test",'rgb'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    return train_dataset,val_dataset,test_dataset









def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}