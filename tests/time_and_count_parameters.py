import fire
from pathlib import Path

#add parrent dir to easy import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#model 
from transformers_video_model.model import VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
#dataloader 
from dataset import get_dataset, collate_fn
from dataset.utils import load_config
#metric funtion
from helper_fn.metric import compute_metrics
from torch.utils.data import DataLoader
from time import time

def main(config_path = "/work/21013187/SAM-SLR-v2/phuoc_src/config/landmarks.yaml"):
    
    config = load_config(config_path)
    #dataset config
    dataset_config  = config['dataset']
    dataset_root_path = dataset_config['dataset_root_path']
    dataset_root_path = Path(dataset_root_path)
    
    #prepare model
    pretrained_path = config['pretrained_path']
    landmark_config = config['landmark']
    from time import time

    for use_landmark in [True, False]:
        for use_landmark_transformers in [True, False]:
            for fusion in ['lately','early']:
                landmark_config['fusion'] = fusion
                landmark_config['use'] = use_landmark
                landmark_config['use_landmark_transformers'] = use_landmark_transformers
                config['landmark'] = landmark_config

                train_dataset, _, _ = get_dataset(dataset_root_path, config=config)
                dataloader = DataLoader(train_dataset, collate_fn=collate_fn)
                
                model = VideoMAEForVideoClassification.from_pretrained(
                    pretrained_path,
                    landmark_config=landmark_config
                )

                if use_landmark:
                    print("Use landmark")
                    if use_landmark_transformers:
                        print("Use transformer landmarks")
                    else:
                        print("Not use transformer landmarks")
                    print(f"Fusion type: {fusion}")
                else:
                    print("Not use landmarks")
                
                # Count total model parameters
                param_total = sum(p.numel() for p in model.parameters())
                print(f"Total params: {param_total}")

                # Measure inference time
                num_test = 10
                data = next(iter(dataloader))
                t_start = time()
                for _ in range(num_test):
                    model(**data)
                t_end = time()
                avg_time = (t_end - t_start) / (config['dataset']['batch_size'] * num_test)
                print(f"Time per sample: {round(avg_time, 4)} s\n")

            
    
if __name__ == '__main__':
    fire.Fire(main)