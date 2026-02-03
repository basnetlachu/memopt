#!/usr/bin/env python3
"""
Example: Automatic optimization with Hugging Face Trainer.

This demonstrates how memopt integrates with HuggingFace's Trainer
using its callback system.
"""

import torch
import torch.nn as nn

try:
    from transformers import (
        Trainer,
        TrainingArguments,
        TrainerCallback,
        PreTrainedModel,
        PretrainedConfig,
    )
except ImportError:
    print("Transformers not installed. Run: pip install transformers")
    exit(1)

from torch.utils.data import Dataset

from memopt.training import TrainingOptimizer, TrainingConfig


# Simple model that follows HuggingFace conventions
class SimpleConfig(PretrainedConfig):
    model_type = "simple"

    def __init__(self, input_size=512, hidden_size=1024, num_labels=10, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels


class SimpleModelForClassification(PreTrainedModel):
    config_class = SimpleConfig

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.classifier(input_ids)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class SimpleDataset(Dataset):
    """Simple dataset for demonstration."""

    def __init__(self, num_samples=1000, input_size=512, num_labels=10):
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, num_labels, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "labels": self.labels[idx],
        }


class MemoptTrainerCallback(TrainerCallback):
    """
    HuggingFace Trainer callback for memopt integration.

    This callback:
    1. Attaches memopt at training start
    2. Monitors training progress via logs
    3. Applies optimizations between epochs
    4. Detaches at training end
    """

    def __init__(self, config: TrainingConfig = None, sample_input: torch.Tensor = None):
        super().__init__()
        self.config = config or TrainingConfig(verbose=True)
        self.sample_input = sample_input
        self.optimizer = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Attach memopt optimizer at training start."""
        self.optimizer = TrainingOptimizer(self.config)
        self.optimizer.attach(model, self.sample_input)
        print("MemOpt: Attached to HuggingFace model")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Detach memopt at training end."""
        if self.optimizer:
            self.optimizer.detach()
            print("MemOpt: Detached from HuggingFace model")
            print(f"MemOpt Stats: {self.optimizer.get_stats()}")

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Notify epoch start."""
        if self.optimizer and self.optimizer._hook:
            epoch = int(state.epoch) if state.epoch else 0
            self.optimizer._hook.notify_epoch_start(epoch)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Notify epoch end - triggers optimization."""
        if self.optimizer and self.optimizer._hook:
            epoch = int(state.epoch) if state.epoch else 0
            self.optimizer._hook.notify_epoch_end(epoch)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Track loss for convergence monitoring."""
        if self.optimizer and self.optimizer._hook and logs:
            loss = logs.get("loss")
            if loss is not None:
                self.optimizer._hook.notify_loss(loss)


def main():
    print("=" * 60)
    print("MEMOPT + HuggingFace Trainer Example")
    print("=" * 60)

    # Create model
    config = SimpleConfig(input_size=512, hidden_size=1024, num_labels=10)
    model = SimpleModelForClassification(config)

    # Create datasets
    train_dataset = SimpleDataset(num_samples=1000)
    eval_dataset = SimpleDataset(num_samples=200)

    # Create sample input for profiling
    sample_input = torch.randn(32, 512)
    if torch.cuda.is_available():
        sample_input = sample_input.cuda()

    # Create memopt callback
    memopt_config = TrainingConfig(
        profile_batches=20,
        optimize_after_epoch=1,
        verbose=True,
    )
    memopt_callback = MemoptTrainerCallback(
        config=memopt_config,
        sample_input=sample_input,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        remove_unused_columns=False,
    )

    # Create trainer with memopt callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[memopt_callback],
    )

    # Train
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
