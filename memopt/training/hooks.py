"""
Training hooks - Hook injection for different ML frameworks.

Supports:
- Pure PyTorch
- PyTorch Lightning
- Hugging Face Trainer
- Accelerate
"""

import logging
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .wrapper import TrainingOptimizer

logger = logging.getLogger("memopt.training")


@dataclass
class HookEvent:
    """Event triggered by a training hook."""
    event_type: str  # epoch_start, epoch_end, batch_start, batch_end, backward, eval_start, eval_end
    epoch: int = 0
    batch: int = 0
    loss: Optional[float] = None
    model: Optional[nn.Module] = None
    extra: Optional[Dict] = None


class TrainingHook(ABC):
    """Base class for training hooks."""

    def __init__(self, optimizer: "TrainingOptimizer"):
        self.optimizer = optimizer
        self._installed = False

    @abstractmethod
    def install(self, target: Any) -> bool:
        """Install hooks on the target (model, trainer, etc.)."""
        pass

    @abstractmethod
    def uninstall(self) -> bool:
        """Remove installed hooks."""
        pass

    def on_event(self, event: HookEvent):
        """Called when a training event occurs."""
        self.optimizer._handle_event(event)


class PyTorchHook(TrainingHook):
    """
    Hooks for pure PyTorch training loops.

    Works by:
    1. Registering backward hooks on the model
    2. Wrapping the forward method
    3. Tracking loss values via monkey-patching
    """

    def __init__(self, optimizer: "TrainingOptimizer"):
        super().__init__(optimizer)
        self._handles: List = []
        self._original_forward = None
        self._original_backward = None
        self._model: Optional[nn.Module] = None
        self._batch_count = 0
        self._epoch = 0

    def install(self, model: nn.Module) -> bool:
        """Install hooks on a PyTorch model."""
        if self._installed:
            return True

        self._model = model

        # Wrap forward to detect batches
        self._original_forward = model.forward
        model.forward = self._wrapped_forward

        # Register backward hooks on parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(
                    self._create_grad_hook(name)
                )
                self._handles.append(handle)

        self._installed = True
        logger.debug(f"Installed PyTorch hooks on {len(self._handles)} parameters")
        return True

    def uninstall(self) -> bool:
        """Remove hooks."""
        if not self._installed:
            return True

        # Restore original forward
        if self._model and self._original_forward:
            self._model.forward = self._original_forward

        # Remove gradient hooks
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        self._installed = False
        return True

    def _wrapped_forward(self, *args, **kwargs):
        """Wrapped forward to detect batch processing."""
        self._batch_count += 1

        # Notify batch start
        self.on_event(HookEvent(
            event_type="batch_start",
            epoch=self._epoch,
            batch=self._batch_count,
            model=self._model,
        ))

        result = self._original_forward(*args, **kwargs)

        return result

    def _create_grad_hook(self, name: str):
        """Create a gradient hook for a parameter."""
        def hook(param):
            # This is called after gradient is accumulated
            pass  # We could track gradients here if needed
        return hook

    def notify_epoch_start(self, epoch: int):
        """Call this at the start of each epoch."""
        self._epoch = epoch
        self._batch_count = 0
        self.on_event(HookEvent(
            event_type="epoch_start",
            epoch=epoch,
            model=self._model,
        ))

    def notify_epoch_end(self, epoch: int, val_loss: Optional[float] = None):
        """Call this at the end of each epoch."""
        self.on_event(HookEvent(
            event_type="epoch_end",
            epoch=epoch,
            loss=val_loss,
            model=self._model,
        ))

    def notify_loss(self, loss: float):
        """Call this after computing loss."""
        self.on_event(HookEvent(
            event_type="loss_computed",
            epoch=self._epoch,
            batch=self._batch_count,
            loss=loss.item() if hasattr(loss, 'item') else loss,
            model=self._model,
        ))


class LightningHook(TrainingHook):
    """
    Hooks for PyTorch Lightning.

    Uses Lightning's callback system.
    """

    def __init__(self, optimizer: "TrainingOptimizer"):
        super().__init__(optimizer)
        self._callback = None

    def install(self, trainer) -> bool:
        """Install callback on Lightning Trainer."""
        if self._installed:
            return True

        try:
            from pytorch_lightning import Callback

            class MemoptCallback(Callback):
                def __init__(self, hook: "LightningHook"):
                    self.hook = hook

                def on_train_epoch_start(self, trainer, pl_module):
                    self.hook.on_event(HookEvent(
                        event_type="epoch_start",
                        epoch=trainer.current_epoch,
                        model=pl_module,
                    ))

                def on_train_epoch_end(self, trainer, pl_module):
                    self.hook.on_event(HookEvent(
                        event_type="epoch_end",
                        epoch=trainer.current_epoch,
                        model=pl_module,
                    ))

                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    loss = outputs.get("loss") if isinstance(outputs, dict) else None
                    self.hook.on_event(HookEvent(
                        event_type="batch_end",
                        epoch=trainer.current_epoch,
                        batch=batch_idx,
                        loss=loss.item() if loss is not None and hasattr(loss, 'item') else loss,
                        model=pl_module,
                    ))

                def on_validation_start(self, trainer, pl_module):
                    self.hook.on_event(HookEvent(
                        event_type="eval_start",
                        epoch=trainer.current_epoch,
                        model=pl_module,
                    ))

                def on_validation_end(self, trainer, pl_module):
                    self.hook.on_event(HookEvent(
                        event_type="eval_end",
                        epoch=trainer.current_epoch,
                        model=pl_module,
                    ))

            self._callback = MemoptCallback(self)
            trainer.callbacks.append(self._callback)
            self._installed = True
            logger.debug("Installed Lightning callback")
            return True

        except ImportError:
            logger.warning("PyTorch Lightning not installed")
            return False

    def uninstall(self) -> bool:
        """Remove callback."""
        # Lightning callbacks are harder to remove cleanly
        self._installed = False
        return True


class HuggingFaceHook(TrainingHook):
    """
    Hooks for Hugging Face Trainer.

    Uses Trainer's callback system.
    """

    def __init__(self, optimizer: "TrainingOptimizer"):
        super().__init__(optimizer)
        self._callback = None

    def install(self, trainer) -> bool:
        """Install callback on HF Trainer."""
        if self._installed:
            return True

        try:
            from transformers import TrainerCallback

            class MemoptTrainerCallback(TrainerCallback):
                def __init__(self, hook: "HuggingFaceHook"):
                    self.hook = hook

                def on_epoch_begin(self, args, state, control, **kwargs):
                    model = kwargs.get("model")
                    self.hook.on_event(HookEvent(
                        event_type="epoch_start",
                        epoch=int(state.epoch) if state.epoch else 0,
                        model=model,
                    ))

                def on_epoch_end(self, args, state, control, **kwargs):
                    model = kwargs.get("model")
                    self.hook.on_event(HookEvent(
                        event_type="epoch_end",
                        epoch=int(state.epoch) if state.epoch else 0,
                        model=model,
                    ))

                def on_log(self, args, state, control, logs=None, **kwargs):
                    model = kwargs.get("model")
                    loss = logs.get("loss") if logs else None
                    self.hook.on_event(HookEvent(
                        event_type="loss_computed",
                        epoch=int(state.epoch) if state.epoch else 0,
                        batch=state.global_step,
                        loss=loss,
                        model=model,
                    ))

                def on_evaluate(self, args, state, control, **kwargs):
                    model = kwargs.get("model")
                    self.hook.on_event(HookEvent(
                        event_type="eval_start",
                        epoch=int(state.epoch) if state.epoch else 0,
                        model=model,
                    ))

            self._callback = MemoptTrainerCallback(self)
            trainer.add_callback(self._callback)
            self._installed = True
            logger.debug("Installed HuggingFace callback")
            return True

        except ImportError:
            logger.warning("Transformers not installed")
            return False

    def uninstall(self) -> bool:
        """Remove callback."""
        self._installed = False
        return True


class AccelerateHook(TrainingHook):
    """
    Hooks for Hugging Face Accelerate.

    Works by wrapping the Accelerator's methods.
    """

    def __init__(self, optimizer: "TrainingOptimizer"):
        super().__init__(optimizer)
        self._accelerator = None
        self._original_backward = None
        self._epoch = 0
        self._step = 0

    def install(self, accelerator) -> bool:
        """Install hooks on Accelerator."""
        if self._installed:
            return True

        self._accelerator = accelerator

        # Wrap backward method
        self._original_backward = accelerator.backward
        accelerator.backward = self._wrapped_backward

        self._installed = True
        logger.debug("Installed Accelerate hooks")
        return True

    def uninstall(self) -> bool:
        """Remove hooks."""
        if self._accelerator and self._original_backward:
            self._accelerator.backward = self._original_backward

        self._installed = False
        return True

    def _wrapped_backward(self, loss, **kwargs):
        """Wrapped backward to track loss."""
        self._step += 1

        self.on_event(HookEvent(
            event_type="loss_computed",
            epoch=self._epoch,
            batch=self._step,
            loss=loss.item() if hasattr(loss, 'item') else float(loss),
        ))

        return self._original_backward(loss, **kwargs)

    def notify_epoch(self, epoch: int):
        """Update current epoch."""
        self._epoch = epoch
        self._step = 0


def detect_framework(target: Any) -> str:
    """
    Detect which ML framework is being used.

    Returns:
        Framework name: "pytorch", "lightning", "huggingface", "accelerate"
    """
    # Check class names
    class_name = type(target).__name__
    module_name = type(target).__module__

    if "lightning" in module_name.lower():
        return "lightning"

    if "transformers" in module_name.lower() and "Trainer" in class_name:
        return "huggingface"

    if "accelerate" in module_name.lower():
        return "accelerate"

    if isinstance(target, nn.Module):
        return "pytorch"

    return "unknown"


def create_hook(optimizer: "TrainingOptimizer", target: Any) -> Optional[TrainingHook]:
    """
    Create appropriate hook for the target.

    Args:
        optimizer: TrainingOptimizer instance
        target: Model, Trainer, or Accelerator

    Returns:
        Appropriate TrainingHook or None
    """
    framework = detect_framework(target)

    if framework == "pytorch":
        return PyTorchHook(optimizer)
    elif framework == "lightning":
        return LightningHook(optimizer)
    elif framework == "huggingface":
        return HuggingFaceHook(optimizer)
    elif framework == "accelerate":
        return AccelerateHook(optimizer)
    else:
        logger.warning(f"Unknown framework for {type(target)}")
        return None
