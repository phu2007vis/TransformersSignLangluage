import fire
from transformers import TrainingArguments, Trainer
from pathlib import Path
from dataset_helper import get_labelmap
from model import VideoMAEForVideoClassification, get_processor
from get_dataloader import get_dataset, collate_fn
from metric import compute_metrics

def main(dataset_root_path= "/work/21013187/SAM-SLR-v2/data/rgb",
         model_ckpt = "MCG-NJU/videomae-base" 
         ,num_epochs: int = 10,
         batch_size: int = 2):
    
    dataset_root_path = Path(dataset_root_path)
    label2id, id2label = get_labelmap(dataset_root_path)
    model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
    image_processor = get_processor(model_ckpt)
    train_dataset, val_dataset, test_dataset = get_dataset(dataset_root_path, model, image_processor)
    
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-finetuned-ucf101-subset"
    
    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-7,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
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
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    train_results = trainer.train()
    trainer.evaluate(test_dataset)
    
    trainer.save_model()
    test_results = trainer.evaluate(test_dataset)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()

if __name__ == '__main__':
    fire.Fire(main)