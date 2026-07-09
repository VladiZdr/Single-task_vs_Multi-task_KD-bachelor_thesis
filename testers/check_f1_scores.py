import os
import sys
from typing import Dict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from configs.model_config import ModelConfig
from fine_tuning.legal_model import LegalModel
from fine_tuning.legal_model_trainer import LegalModelTrainer
from fine_tuning.train_legal_model import prepare_dataloaders, models_to_run


def _validate_metrics(metrics: Dict[str, float]) -> None:
    expected_keys = {"loss", "macro_f1", "micro_f1"}
    assert expected_keys <= set(metrics), f"Missing metric keys: {expected_keys - set(metrics)}"
    assert metrics["loss"] >= 0.0, "Loss should be non-negative"
    assert 0.0 <= metrics["macro_f1"] <= 1.0, "Macro-F1 must be in [0, 1]"
    assert 0.0 <= metrics["micro_f1"] <= 1.0, "Micro-F1 must be in [0, 1]"


@torch.no_grad()
def evaluate_model(param_config: ModelConfig) -> Dict[str, float]:
    current_config = param_config
    checkpoint_path = os.path.join(current_config.checkpoint_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint for {current_config.task_name}_{current_config.unique_id_for_dir}: {checkpoint_path}")

    train_loader, val_loader, test_loader = prepare_dataloaders(task_config=current_config)
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Validation loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"

    model = LegalModel(current_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(current_config.device)))

    trainer = LegalModelTrainer(model, current_config)
    trainer._remove_teacher_weight_for_evaluation()  # Ensure teacher weight is set to 0 for evaluation
    metrics = trainer.evaluate(val_loader)
    _validate_metrics(metrics)

    print(f"Evaluation metrics for {current_config.task_name}_{current_config.unique_id_for_dir}: {metrics}")
    return metrics

def check_all_f1_scores() -> None:
    for config in models_to_run:
        evaluate_model(param_config=config)

    print("\nAll F1 checks passed.")

if __name__ == "__main__":
    check_all_f1_scores()
