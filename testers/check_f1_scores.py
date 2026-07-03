import os
import sys
from typing import Dict, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import configs.model_config as model_config
from configs.model_config import ModelConfig
from fine_tuning.legal_model import LegalModel
from fine_tuning.legal_model_trainer import LegalModelTrainer
from fine_tuning.train_legal_model import prepare_dataloaders


def _build_ledgar_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            task_name="ledgar",
            num_labels=100,
            problem_type="single_label",
            loss_type="cross_entropy",
            model_name_or_path="nlpaueb/legal-bert-base-uncased",
            num_of_batches=-1,
            percent_of_data=100,
            batch_size=8,
            learning_rate=3e-5,
            epochs=5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            T=1.0,
            alpha=0.5,
            loss_reduction="mean",
            kd_teacher_weight_schedule="constant",
            kd_teacher_weight_start=1.0,
            kd_teacher_weight_end=1.0,
            device="auto",
            seed=42,
            checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher",
            output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
            unique_id_for_dir="Teacher",
            preprocessed_data_dir="raw",
        ),
        ModelConfig(
            task_name="ledgar",
            num_labels=100,
            problem_type="single_label",
            loss_type="cross_entropy",
            model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
            num_of_batches=-1,
            percent_of_data=100,
            batch_size=8,
            learning_rate=3e-5,
            epochs=5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            T=1.0,
            alpha=0.5,
            loss_reduction="mean",
            kd_teacher_weight_schedule="constant",
            kd_teacher_weight_start=1.0,
            kd_teacher_weight_end=1.0,
            device="auto",
            seed=42,
            checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student",
            output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_outputs",
            unique_id_for_dir="Baseline",
            preprocessed_data_dir="raw",
        ),
        ModelConfig(
            task_name="ledgar",
            num_labels=100,
            problem_type="single_label",
            loss_type="kldiv",
            model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
            num_of_batches=-1,
            percent_of_data=100,
            batch_size=8,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            T=1.0,
            alpha=0.5,
            loss_reduction="mean",
            kd_teacher_weight_schedule="linear_epoch",
            kd_teacher_weight_start=1.0,
            kd_teacher_weight_end=0.0,
            device="auto",
            seed=42,
            checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student",
            output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_outputs",
            unique_id_for_dir="Single_task_KD_Student",
            preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
        ),
    ]


def _build_unfair_tos_configs() -> list[ModelConfig]:
    return [
        model_config.unfair_tos_teacher_tester,
        model_config.unfair_tos_supervised_student_tester,
        model_config.unfair_tos_kd_student_tester,
    ]


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
    metrics = trainer.evaluate(val_loader)
    _validate_metrics(metrics)

    print(f"Evaluation metrics for {current_config.task_name}_{current_config.unique_id_for_dir}: {metrics}")
    return metrics


def _all_configs() -> Iterable[ModelConfig]:
    yield from _build_unfair_tos_configs()
    #yield from _build_ledgar_configs()


def main() -> None:
    for config in _all_configs():
        evaluate_model(param_config=config)

    print("\nAll F1 checks passed.")


if __name__ == "__main__":
    main()
