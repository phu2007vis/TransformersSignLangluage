import fire
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
from dataset.utils import load_config
#metric funtion
from helper_fn.metric import compute_metrics


def main(config_path = "/work/21013187/SAM-SLR-v2/phuoc_src/config/landmarks.yaml" ):
    
    config = load_config(config_path)
    #dataset config
    dataset_config  = config['dataset']
    dataset_root_path = dataset_config['dataset_root_path']
    batch_size = dataset_config['batch_size']
    
    #train config
    train_config = config['train']
    num_epochs = train_config['num_epochs']
    learning_rate = train_config['learning_rate']
   
    #prepare model
    pretrained_path = config['pretrained_path']
    landmark_config = config['landmark']
    model = VideoMAEForVideoClassification.from_pretrained(pretrained_path,landmark_config = landmark_config)
 
    #get all dataset
    dataset_root_path = Path(dataset_root_path)
    train_dataset, val_dataset, test_dataset = get_dataset(dataset_root_path)
    
    #save dir
    model_name = pretrained_path.split("/")[-1]
    new_model_name = f"{model_name}-finetuned"
    
    print(f"Batch size: {batch_size}") 
    print(f"Epochs : {num_epochs}")
    
    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=8,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        report_to="none",
        fp16 = False
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        
    )
    
    trainer.train()

    #evaluate after trainnign
    test_results = trainer.evaluate(test_dataset)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_model()
    trainer.save_state()

if __name__ == '__main__':
    fire.Fire(main)