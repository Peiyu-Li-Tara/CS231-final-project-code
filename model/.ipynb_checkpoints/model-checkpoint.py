##### imports #####

import itertools
import json
import os

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Create a mock sctypes dictionary
np.sctypes = {
    "float": [np.float16, np.float32, np.float64],
    "int": [np.int8, np.int16, np.int32, np.int64],
    "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
    "complex": [np.complex64, np.complex128]
}

# import wandb  # Optional: uncomment if you want to use wandb


##### meta data ##### 
INPUT_PATH = "/home/ubuntu/CS231N/APKLOT_CS231N"
OUTPUT_PATH = "/home/ubuntu/CS231N/outputs_enhanced"  # NEW OUTPUT PATH
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"Results will be saved to: {OUTPUT_PATH}")

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

NUM_PROC = 1
CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"

##### helpers: data processing ##### 
def audit_data(ds, amt=1000):
    ds_copy = ds.select(range(amt))
    
    bad_image_count = 0
    bad_mask_count = 0
    i = 0
    for ex in ds_copy:
        img = ex.get("image")
        msk = ex.get("mask")
    
        is_bad_image = not isinstance(img, Image.Image)
        is_bad_mask = not isinstance(msk, Image.Image)
    
        if is_bad_image:
            bad_image_count += 1
        if is_bad_mask:
            bad_mask_count += 1
        if i % 100 == 0:
            print(i)
        i += 1
    
    print(f"Bad images: {bad_image_count} / {amt}")
    print(f"Bad masks: {bad_mask_count} / {amt}")

def fetch_and_open(example):
    try:
        img = Image.open(example["image"]).convert("RGB")
        msk = Image.open(example["mask"]).convert("L")
        return {"image": img, "mask": msk}
    except Exception:
        return {}

def show_image(image, mask, color=(255,0,0), alpha=0.5):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    colored_mask = Image.new("RGBA", image.size, color + (0,))
    def make_alpha(p):
        return int((p > 0) * 255 * alpha)
    alpha_channel = mask.point(make_alpha)
    colored_mask.putalpha(alpha_channel)
    return Image.alpha_composite(image, colored_mask)

def is_valid_example(example):
    img = example.get("image", None)
    msk = example.get("mask", None)
    
    if not isinstance(img, Image.Image):
        return False
    if not isinstance(msk, Image.Image):
        return False

    return True

##### helpers: eval pipeline ##### 
@dataclass
class ExperimentConfig:
    experiment_name: str
    model_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 2
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Training configuration
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Data configuration
    train_data_fraction: float = 1.0  # Use subset for quick experiments
    eval_data_fraction: float = 1.0
    
    def to_dict(self):
        return asdict(self)

class ExperimentTracker:
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        self.results_file = os.path.join(base_output_dir, "experiment_results.json")
        self.results = self.load_results()
    
    def load_results(self) -> List[Dict]:
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_experiment(self, config: ExperimentConfig, metrics: Dict[str, Any], 
                       model_path: str, training_history: List[Dict]):
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": config.experiment_name,
            "config": config.to_dict(),
            "final_metrics": metrics,
            "model_path": model_path,
            "training_history": training_history
        }
        
        self.results.append(experiment)
        
        # Save to file
        os.makedirs(self.base_output_dir, exist_ok=True)
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Experiment '{config.experiment_name}' saved!")
    
    def get_best_experiment(self, metric: str = "eval_accuracy") -> Optional[Dict]:
        if not self.results:
            return None
        
        best = max(self.results, 
                  key=lambda x: x.get("final_metrics", {}).get(metric, 0))
        return best
    
    def compare_experiments(self, metric: str = "eval_accuracy") -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for exp in self.results:
            row = {
                "experiment_name": exp["experiment_name"],
                "timestamp": exp["timestamp"],
                **exp["config"],
                **exp.get("final_metrics", {})
            }
            data.append(row)
        
        return pd.DataFrame(data)

class MetricsTracker(TrainerCallback):
    def __init__(self):
        self.training_history = []
        self.current_epoch_logs = {}
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs:
            # Normalize training loss key
            logs = logs.copy()
            if "loss" in logs and "train_loss" not in logs:
                logs["train_loss"] = logs.pop("loss")
                
            step_log = {
                "step": state.global_step,
                "epoch": state.epoch,
                **logs
            }
            self.training_history.append(step_log)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called during evaluation to log eval metrics."""
        if metrics:
            eval_log = {
                "step": state.global_step,
                "epoch": state.epoch,
                **metrics
            }
            self.training_history.append(eval_log)
    
    def get_history_df(self) -> pd.DataFrame:
        """Get training history as DataFrame"""
        return pd.DataFrame(self.training_history)

def create_hyperparameter_grid(**param_grids) -> List[Dict]:
    keys = list(param_grids.keys())
    values = list(param_grids.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations

##### helpers: viz functions ##### 
def plot_training_curves(history_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Loss curves
    train_loss = history_df[history_df['train_loss'].notna()]
    eval_loss = history_df[history_df['eval_loss'].notna()]
    
    axes[0, 0].plot(train_loss['step'], train_loss['train_loss'], 
                    label='Train Loss', alpha=0.7)
    axes[0, 0].plot(eval_loss['step'], eval_loss['eval_loss'], 
                    label='Eval Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    eval_acc = history_df[history_df['eval_accuracy'].notna()]
    axes[0, 1].plot(eval_acc['step'], eval_acc['eval_accuracy'], 
                    label='Eval Accuracy', color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU curves
    eval_iou = history_df[history_df['eval_iou'].notna()]
    axes[1, 0].plot(eval_iou['step'], eval_iou['eval_iou'], 
                    label='Overall IoU', color='green', alpha=0.7)
    if 'eval_iou_parking' in history_df.columns:
        eval_iou_parking = history_df[history_df['eval_iou_parking'].notna()]
        axes[1, 0].plot(eval_iou_parking['step'], eval_iou_parking['eval_iou_parking'], 
                        label='Parking IoU', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('IoU Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'learning_rate' in history_df.columns:
        lr_data = history_df[history_df['learning_rate'].notna()]
        axes[1, 1].plot(lr_data['step'], lr_data['learning_rate'], 
                        label='Learning Rate', color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def extract_results(experiments: dict, metrics: dict):
    raw_experiment_names = list(experiments.keys())
    mapped_experiment_names = list(experiments.values())
    raw_metric_names = list(metrics.keys())
    mapped_metric_names = list(metrics.values())
    df = pd.DataFrame(index=mapped_experiment_names, columns=mapped_metric_names)
    with open(os.path.join(OUTPUT_PATH, 'experiment_results.json'), 'r') as f:
        all_results = json.load(f)
    already_done = {}
    for i in range(len(all_results)):
        # iterate through all experiments 
        these_results = all_results[i]
        this_time_stamp = these_results['timestamp']
        this_raw_name = these_results['experiment_name']

        # check if we already have this run, and existing is "newer"
        if this_raw_name in already_done:
            if pd.Timestamp(this_time_stamp) < pd.Timestamp(already_done[this_raw_name]):
                continue
        # check if this run in desired runs 
        if this_raw_name in raw_experiment_names:
            this_mapped_name = experiments[this_raw_name]
            for this_raw_metric in raw_metric_names:
                df.loc[this_mapped_name, metrics[this_raw_metric]] = these_results['final_metrics'][this_raw_metric]
            already_done[this_raw_name] = this_time_stamp
    return df

def plot_experiment_comparison(df, add_annotations=True, y_axis_range=(0, 1), subplot_height=5, subplot_width=6):
    """Plot comparison of different experiments"""
    metrics = list(df.columns)
    fig, axes = plt.subplots(1, len(metrics), figsize=(subplot_width * len(metrics), subplot_height))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            bars = axes[i].bar(range(len(df)), df[metric])
            axes[i].set_ylabel(metric)
            axes[i].set_title(f"{metric} Comparison")
            axes[i].set_xticks(range(len(df)))
            axes[i].set_xticklabels(list(df.index), rotation=45, ha='right')
            axes[i].set_ylim(y_axis_range)
            
            # Add value labels on bars
            if add_annotations:
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    return plt

def visualize_predictions(model, processor, test_images: List, 
                         test_masks: List = None, num_samples: int = 4):
    """Visualize model predictions on test images"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3 if test_masks else 2, 
                            figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(test_images))):
        image = test_images[i]
        
        # Preprocess image
        if test_masks:
            processed = processor({"image": image, "mask": test_masks[i]})
        else:
            # Create dummy mask for preprocessing
            dummy_mask = Image.new("L", image.size, color=0)
            processed = processor({"image": image, "mask": dummy_mask})
        
        pixel_values = processed["pixel_values"].unsqueeze(0).to(model.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=image.size[::-1], 
                mode="bilinear", align_corners=False
            )
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
        
        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Plot prediction
        pred_overlay = show_image(image, pred_mask_pil, color=(255,0,0), alpha=0.4)
        axes[i, 1].imshow(pred_overlay)
        axes[i, 1].set_title(f'Prediction {i+1}')
        axes[i, 1].axis('off')
        
        # Plot ground truth if available
        if test_masks:
            gt_overlay = show_image(image, test_masks[i], color=(0,255,0), alpha=0.4)
            axes[i, 2].imshow(gt_overlay)
            axes[i, 2].set_title(f'Ground Truth {i+1}')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    return plt

##### load and process data ##### 
ds = load_dataset(
    f"{INPUT_PATH}/APKLOT_CS231N.py",
    name="default",
    split="train",
    data_dir=INPUT_PATH,
)

print(f"Dataset size: {len(ds)}")

splits = ds.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

print(f"Training samples: {len(train_ds)}")
print(f"Evaluation samples: {len(eval_ds)}")

train_ds = train_ds.filter(
    is_valid_example,
    num_proc=NUM_PROC,
    load_from_cache_file=False
)

eval_ds = eval_ds.filter(
    is_valid_example,
    num_proc=NUM_PROC,
    load_from_cache_file=False
)

print(f"After filtering - Training: {len(train_ds)}, Evaluation: {len(eval_ds)}")

processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

def preprocess(example):
    img = example.get("image")
    msk = example.get("mask")
    # Validate image and mask
    if not isinstance(img, Image.Image):
        print("Bad image")
        img = Image.new("RGB", (512, 512), color=(0, 0, 0))
    if not isinstance(msk, Image.Image):
        print("Bad mask")
        msk = Image.new("L", (512, 512), color=0)

    if img.mode != "RGB":
        img = img.convert("RGB")
    if msk.mode != "L":
        msk = msk.convert("L")
    
    # now the processor will see a clean array and infer channels automatically
    encoding = processor(
        images=img,
        segmentation_maps=msk,
        return_tensors="pt",
    )
    
    pixel_values = encoding.get("pixel_values")
    labels = encoding.get("labels")

    if not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    labels = torch.where(labels == 255, 0, 1).long()
    
    return {
        "pixel_values": pixel_values.squeeze(0),  # shape: [C, H, W]
        "labels": labels.squeeze(0),              # shape: [H, W]
    }

# Apply processing
print("Preprocessing training data...")
train_ds = train_ds.map(
    preprocess,
    batched=False,
    num_proc=NUM_PROC,
    remove_columns=["image", "mask"],
    load_from_cache_file=False
)

print("Preprocessing evaluation data...")
eval_ds = eval_ds.map(
    preprocess,
    batched=False,
    num_proc=NUM_PROC,
    remove_columns=["image", "mask"],
    load_from_cache_file=False
)

print("Preprocessing complete!")

##### make model ##### 

id2label = {
    0: "background",
    1: "parking_lot"
}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

print(f"Labels: {id2label}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy_metric = MulticlassAccuracy(num_classes=2, average='micro').to(device)
f1_metric = MulticlassF1Score(num_classes=2, average='micro').to(device)
iou_metric = MulticlassJaccardIndex(num_classes=2, average='macro').to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors and move to device
    preds = torch.from_numpy(np.argmax(logits, axis=1)).to(device)
    labels = torch.from_numpy(labels).to(device)
    
    # Interpolate predictions to match label size
    preds = F.interpolate(
        preds.unsqueeze(1).float(),
        size=labels.shape[-2:], 
        mode="nearest"
    ).squeeze(1).long()
    
    # Flatten for pixel-wise metrics
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    # Reset and compute metrics
    accuracy_metric.reset()
    f1_metric.reset()
    iou_metric.reset()
    
    accuracy = accuracy_metric(preds_flat, labels_flat).item()
    f1 = f1_metric(preds_flat, labels_flat).item()
    iou = iou_metric(preds_flat, labels_flat).item()
    
    # Additional metrics
    # Class-wise IoU
    intersection_0 = ((preds_flat == 0) & (labels_flat == 0)).sum().float()
    union_0 = ((preds_flat == 0) | (labels_flat == 0)).sum().float()
    iou_background = (intersection_0 / union_0).item() if union_0 > 0 else 0.0
    
    intersection_1 = ((preds_flat == 1) & (labels_flat == 1)).sum().float()
    union_1 = ((preds_flat == 1) | (labels_flat == 1)).sum().float()
    iou_parking = (intersection_1 / union_1).item() if union_1 > 0 else 0.0
    
    # Pixel counts for analysis
    total_pixels = labels_flat.numel()
    parking_pixels = (labels_flat == 1).sum().item()
    parking_ratio = parking_pixels / total_pixels
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "iou": iou,
        "iou_background": iou_background,
        "iou_parking": iou_parking,
        "parking_pixel_ratio": parking_ratio,
        "total_pixels": total_pixels
    }

def collate_fn(batch):
    pixel_values_batch = []
    labels_batch = []

    for i, ex in enumerate(batch):
        pixel = ex["pixel_values"]
        label = ex["labels"]

        # make sure both are torch.Tensor
        if not isinstance(pixel, torch.Tensor):
            pixel = torch.tensor(pixel, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        pixel_values_batch.append(pixel)
        labels_batch.append(label)

    return {
        "pixel_values": torch.stack(pixel_values_batch),
        "labels": torch.stack(labels_batch),
    }
    
##### training pipeline #####

def run_experiment(config: ExperimentConfig, 
                        train_dataset, eval_dataset, 
                        tracker: ExperimentTracker,
                        use_wandb: bool = False) -> Dict[str, Any]:
    """Run experiment with fixed metrics tracking"""
    
    print(f"\n{'='*50}")
    print(f"Running Experiment: {config.experiment_name}")
    print(f"{'='*50}")
    
    # Create experiment output directory
    experiment_output_dir = os.path.join(
        tracker.base_output_dir, 
        f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    # Load model
    model = AutoModelForSemanticSegmentation.from_pretrained(
        config.model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Subset training data if requested
    if config.train_data_fraction < 1.0:
        subset_size = int(len(train_dataset) * config.train_data_fraction)
        train_dataset = train_dataset.select(range(subset_size))
        print(f"Using {subset_size} training samples ({config.train_data_fraction:.1%})")

    if config.eval_data_fraction < 1.0:
        subset_size = int(len(eval_dataset) * config.eval_data_fraction)
        eval_dataset = eval_dataset.select(range(subset_size))
        print(f"Using {subset_size} evaluation samples ({config.eval_data_fraction:.1%})") 
    
    # Setup FIXED metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training arguments with more frequent logging
    training_args = TrainingArguments(
        output_dir=experiment_output_dir,
        overwrite_output_dir=True,
        run_name=config.experiment_name,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        fp16=True,
        dataloader_num_workers=NUM_PROC,
        dataloader_pin_memory=True,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=min(50, config.logging_steps),  # More frequent logging
        report_to=["wandb"] if use_wandb else ["none"],
        logging_dir=os.path.join(experiment_output_dir, "logs"),
        disable_tqdm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[metrics_tracker]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Final evaluation
    print("Running final evaluation...")
    final_metrics = trainer.evaluate()
    
    # Get training history
    history_df = metrics_tracker.get_history_df()
    
    # Save results
    tracker.save_experiment(
        config=config,
        metrics=final_metrics,
        model_path=experiment_output_dir,
        training_history=history_df.to_dict('records')
    )
    
    # Plot training curves with FIXED plotting function
    plot_save_path = os.path.join(experiment_output_dir, "training_curves.png")
    training_curve = plot_training_curves(history_df, save_path=plot_save_path)
    training_curve.show()
    
    # Save model
    trainer.save_model()
    
    print(f"Experiment completed! Results saved to: {experiment_output_dir}")
    print(f"Final metrics: {final_metrics}")
    
    return {
        "config": config,
        "metrics": final_metrics,
        "model_path": experiment_output_dir,
        "history": history_df
    }

def run_hyperparameter_search(base_config: ExperimentConfig,
                             param_grid: Dict[str, List],
                             train_dataset, eval_dataset,
                             tracker: ExperimentTracker,
                             max_experiments: int = None) -> List[Dict]:
    """Run hyperparameter search"""
    
    combinations = create_hyperparameter_grid(**param_grid)
    
    if max_experiments:
        combinations = combinations[:max_experiments]
    
    print(f"Running {len(combinations)} hyperparameter experiments...")
    
    results = []
    for i, params in enumerate(combinations):
        config_dict = base_config.to_dict()
        config_dict.pop("experiment_name", None) 
        
        # Create experiment config
        config = ExperimentConfig(
            experiment_name=f"{base_config.experiment_name}_hp_{i+1:03d}",
            **{**config_dict, **params}
        )
        
        try:
            result = run_experiment(config, train_dataset, eval_dataset, tracker)
            results.append(result)
        except Exception as e:
            print(f"Experiment {config.experiment_name} failed: {str(e)}")
            continue
    
    return results
