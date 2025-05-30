
from pathlib import Path

#add parrent dir to easy import
import sys
import os
sys.path.append(os.path.dirname(__file__))
#model 
from transformers_video_model.model import VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
#dataloader 
from dataset import get_dataset, collate_fn
from pathlib import Path
#metric funtion
from helper_fn.metric import compute_metrics
from dataset.utils import load_config

def main( ):
    #trans_lately
    # config_path = "/work/21013187/SAM-SLR-v2/phuoc_src/config/landmarks.yaml"
    # pretrained_path ="/work/21013187/SAM-SLR-v2/phuoc_src/trans_lately/checkpoint-30000"
    #rgb only
    pretrained_path = "/work/21013187/SAM-SLR-v2/phuoc_src/rgb_only/checkpoint-30000"
    config_path = "/work/21013187/SAM-SLR-v2/phuoc_src/config/rgb_only.yaml"
    config = load_config(config_path)
    #dataset config
    dataset_config  = config['dataset']
    dataset_root_path = Path(dataset_config['dataset_root_path'])
    batch_size = dataset_config['batch_size']
    
    #train config
    train_config = config['train']
    num_epochs = train_config['num_epochs']
    learning_rate = train_config['learning_rate']
   
    #prepare model
    
    landmark_config = config['landmark']
    model = VideoMAEForVideoClassification.from_pretrained(pretrained_path,landmark_config = landmark_config)
    
    #get all dataset
    train_dataset, val_dataset, test_dataset = get_dataset(dataset_root_path,config = config)
  
    args = TrainingArguments(
        "testing",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        report_to="none",
        
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
   

    test_results = trainer.evaluate(test_dataset)
    trainer.log_metrics("rgb", test_results)
    trainer.save_metrics("rgb", test_results)


if __name__ == '__main__':
    main()