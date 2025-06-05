from model import (
    ExperimentTracker, ExperimentConfig, run_hyperparameter_search, 
    train_ds, eval_ds, OUTPUT_PATH
)

# Initialize tracker
tracker = ExperimentTracker(OUTPUT_PATH)

# Base configuration
base_config = ExperimentConfig(
    experiment_name="scale_1.0",
    logging_steps=250 * 4,
    eval_steps=250 * 4,
)

# Define hyperparameter grid
param_grid = {
    'learning_rate': [5e-4],
    'per_device_train_batch_size': [8],
    'weight_decay': [0.01],
    'warmup_steps': [100 * 4],
    'num_train_epochs': [1],
    'train_data_fraction': [1.0],
    'eval_data_fraction': [1.0],
}

# Run search
results = run_hyperparameter_search(
    base_config=base_config,
    param_grid=param_grid,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tracker=tracker,
)

print("Hyperparameter search completed!")