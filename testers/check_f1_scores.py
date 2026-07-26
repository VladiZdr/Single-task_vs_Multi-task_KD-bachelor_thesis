import os
import sys
import time
from math import isfinite
from typing import Any, Dict, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from configs.model_config import ModelConfig, MultiTaskModelConfig
from fine_tuning.legal_model import LegalModel
from fine_tuning.legal_model_trainer import LegalModelTrainer
from fine_tuning.train_legal_model import prepare_dataloaders, models_to_run as single_task_models_to_run
from multi_task.multi_task_model import MultiTaskModel
from multi_task.multi_task_trainer import MultiTaskTrainer
from multi_task.train_multi_task import prepare_multitask_dataloaders

def _format_label_str(per_label_dict: Dict[str, Dict[str, float]], find_max: bool) -> str:
    if not per_label_dict:
        return "-"
    if find_max:
        label, data = max(per_label_dict.items(), key=lambda item: item[1]["f1"])
    else:
        label, data = min(per_label_dict.items(), key=lambda item: item[1]["f1"])
    return f"{label} ({data['f1']:.2f})"

def print_table(single_results, multi_results):
    #  Print Final Summary Table 
    header_len = 140
    print("\n" + "=" * header_len)
    print(" FINAL EVALUATION SUMMARY ".center(header_len, "="))
    print("=" * header_len)
    print(f"{'Model / Task Sub-Split':<50} | {'Loss':<7} | {'Macro-F1':<8} | {'Micro-F1':<8} | {'Best Label (F1)':<18} | {'Worst Label (F1)':<18} | {'Throughput':<10}")
    print("-" * header_len)

    # Print Single-Task Models
    for name, res in single_results:
        m = res["metrics"]
        a = res.get("analysis", {})
        per_label = a.get("per_label", {})
        eff = a.get("efficiency", {})

        best_lbl = _format_label_str(per_label, find_max=True)
        worst_lbl = _format_label_str(per_label, find_max=False)
        throughput = f"{eff.get('samples_per_second', 0.0):.1f} smp/s" if eff else "-"

        print(f"{name:<50} | {m.get('loss', 0.0):<7.4f} | {m.get('macro_f1', 0.0):<8.4f} | {m.get('micro_f1', 0.0):<8.4f} | {best_lbl:<18} | {worst_lbl:<18} | {throughput:<10}")

    # Print Multi-Task Models
    for name, res in multi_results:
        m = res["metrics"]
        analyses = res.get("analyses", {})

        # Overall row
        print(f"{'[MultiTask] ' + name:<50} | {m.get('loss', 0.0):<7.4f} | {m.get('macro_f1', 0.0):<8.4f} | {m.get('micro_f1', 0.0):<8.4f} | {'-':<18} | {'-':<18} | {'-':<10}")

        # Sub-tasks breakdown
        for task_key in ["ledgar", "unfair_tos"]:
            t_loss = m.get(f"{task_key}_loss", 0.0)
            t_macro = m.get(f"{task_key}_macro_f1", 0.0)
            t_micro = m.get(f"{task_key}_micro_f1", 0.0)

            task_ana = analyses.get(task_key, {})
            per_label = task_ana.get("per_label", {})
            eff = task_ana.get("efficiency", {})

            best_lbl = _format_label_str(per_label, find_max=True)
            worst_lbl = _format_label_str(per_label, find_max=False)
            throughput = f"{eff.get('samples_per_second', 0.0):.1f} smp/s" if eff else "-"

            prefix = "  ├─ LEDGAR" if task_key == "ledgar" else "  └─ UNFAIR-ToS"
            print(f"{prefix:<50} | {t_loss:<7.4f} | {t_macro:<8.4f} | {t_micro:<8.4f} | {best_lbl:<18} | {worst_lbl:<18} | {throughput:<10}")

    print("=" * header_len + "\n")
    print("All F1 checks passed.")

def verify_metrics(metrics: Dict[str, float], required_keys: Iterable[str] = ("loss", "macro_f1", "micro_f1")) -> None:
    expected_keys = set(required_keys)
    missing_keys = expected_keys - set(metrics)
    assert not missing_keys, f"Missing metric keys: {missing_keys}"

    for key in expected_keys:
        value = metrics[key]
        assert isinstance(value, (int, float)), f"Metric '{key}' must be numeric, got {type(value)!r}"
        assert isfinite(float(value)), f"Metric '{key}' must be finite, got {value}"

    assert metrics["loss"] >= 0.0, "Loss should be non-negative"
    assert 0.0 <= metrics["macro_f1"] <= 1.0, "Macro-F1 must be in [0, 1]"
    assert 0.0 <= metrics["micro_f1"] <= 1.0, "Micro-F1 must be in [0, 1]"


def _collect_task_predictions(trainer: MultiTaskTrainer, dataloader, task_name: str):
    all_preds = []
    all_labels = []
    start_time = time.perf_counter()
    batch_count = 0

    for batch in dataloader:
        prepared = trainer._prepare_batch(batch)
        labels = prepared["labels"]
        task = prepared["task"]

        assert isinstance(labels, torch.Tensor)
        assert isinstance(task, str)
        assert task == task_name, f"Expected task '{task_name}', got '{task}'"

        logits = trainer.model(prepared["input_ids"], prepared["attention_mask"], prepared["token_type_ids"], task=task)

        if trainer.task_configs[task].problem_type == "multi_label":
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        else:
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())
        batch_count += 1

    elapsed_seconds = time.perf_counter() - start_time

    if not all_preds:
        raise ValueError(f"No batches were processed for task '{task_name}'")

    preds_array = np.concatenate(all_preds, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    return labels_array, preds_array, batch_count, elapsed_seconds


def _collect_single_task_predictions(trainer: LegalModelTrainer, dataloader):
    all_preds = []
    all_labels = []
    start_time = time.perf_counter()
    batch_count = 0

    for batch in dataloader:
        prepared = trainer._prepare_batch(batch)
        labels = prepared["labels"]

        assert isinstance(labels, torch.Tensor)

        logits = trainer.model(prepared["input_ids"], prepared["attention_mask"], prepared["token_type_ids"])
        if trainer.config.problem_type == "multi_label":
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        else:
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())
        batch_count += 1

    elapsed_seconds = time.perf_counter() - start_time

    if not all_preds:
        raise ValueError("No batches were processed for single-task analysis")

    preds_array = np.concatenate(all_preds, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    return labels_array, preds_array, batch_count, elapsed_seconds


def analyze_single_task_performance_and_efficiency(
    trainer: LegalModelTrainer,
    dataloader,
    task_name: str,
    num_labels: int,
) -> Dict[str, Dict[str, float]]:
    labels_array, preds_array, batch_count, elapsed_seconds = _collect_single_task_predictions(trainer, dataloader)

    if labels_array.ndim == 1:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_array,
            preds_array,
            labels=list(range(num_labels)),
            average=None,
            zero_division=0,
        )
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_array,
            preds_array,
            average=None,
            zero_division=0,
        )

    per_label_metrics: Dict[str, Dict[str, float]] = {}
    for index in range(num_labels):
        per_label_metrics[f"label_{index}"] = {
            "precision": float(precision[index]),   #type: ignore
            "recall": float(recall[index]),         #type: ignore
            "f1": float(f1[index]),                 #type: ignore
            "support": float(support[index]),       #type: ignore
        }

    total_samples = int(labels_array.shape[0])
    efficiency = {
        "elapsed_seconds": float(elapsed_seconds),
        "num_batches": float(batch_count),
        "num_samples": float(total_samples),
        "samples_per_second": float(total_samples / elapsed_seconds) if elapsed_seconds > 0 else float("inf"),
        "milliseconds_per_sample": float((elapsed_seconds / total_samples) * 1000.0) if total_samples > 0 else 0.0,
        "milliseconds_per_batch": float((elapsed_seconds / batch_count) * 1000.0) if batch_count > 0 else 0.0,
    }

    print(
        f"Per-label analysis for {task_name}: "
        f"best={max(per_label_metrics.items(), key=lambda item: item[1]['f1'])[0]} "
        f"worst={min(per_label_metrics.items(), key=lambda item: item[1]['f1'])[0]} "
        f"throughput={efficiency['samples_per_second']:.2f} samples/s"
    )

    return {
        "per_label": per_label_metrics,  #type: ignore
        "efficiency": efficiency,
    }


def analyze_per_label_performance_and_efficiency(
    trainer: MultiTaskTrainer,
    dataloader,
    task_name: str,
    num_labels: int,
) -> Dict[str, Dict[str, float]]:
    labels_array, preds_array, batch_count, elapsed_seconds = _collect_task_predictions(trainer, dataloader, task_name)

    if labels_array.ndim == 1:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_array,
            preds_array,
            labels=list(range(num_labels)),
            average=None,
            zero_division=0,
        )
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_array,
            preds_array,
            average=None,
            zero_division=0,
        )

    per_label_metrics: Dict[str, Dict[str, float]] = {}
    for index in range(num_labels):
        per_label_metrics[f"label_{index}"] = {
            "precision": float(precision[index]),   #type: ignore
            "recall": float(recall[index]),         #type: ignore
            "f1": float(f1[index]),                 #type: ignore
            "support": float(support[index]),       #type: ignore
        }

    total_samples = int(labels_array.shape[0])
    efficiency = {
        "elapsed_seconds": float(elapsed_seconds),
        "num_batches": float(batch_count),
        "num_samples": float(total_samples),
        "samples_per_second": float(total_samples / elapsed_seconds) if elapsed_seconds > 0 else float("inf"),
        "milliseconds_per_sample": float((elapsed_seconds / total_samples) * 1000.0) if total_samples > 0 else 0.0,
        "milliseconds_per_batch": float((elapsed_seconds / batch_count) * 1000.0) if batch_count > 0 else 0.0,
    }

    assert len(per_label_metrics) == num_labels, f"Expected {num_labels} label metrics, got {len(per_label_metrics)}"
    for label_name, metrics in per_label_metrics.items():
        assert 0.0 <= metrics["precision"] <= 1.0, f"{label_name} precision must be in [0, 1]"
        assert 0.0 <= metrics["recall"] <= 1.0, f"{label_name} recall must be in [0, 1]"
        assert 0.0 <= metrics["f1"] <= 1.0, f"{label_name} F1 must be in [0, 1]"
        assert metrics["support"] >= 0.0, f"{label_name} support must be non-negative"

    assert efficiency["elapsed_seconds"] >= 0.0
    assert efficiency["num_batches"] > 0.0
    assert efficiency["num_samples"] > 0.0
    assert efficiency["samples_per_second"] > 0.0

    best_label = max(per_label_metrics.items(), key=lambda item: item[1]["f1"])
    worst_label = min(per_label_metrics.items(), key=lambda item: item[1]["f1"])
    print(
        f"Per-label analysis for {task_name}: "
        f"best={best_label[0]} (F1={best_label[1]['f1']:.4f}), "
        f"worst={worst_label[0]} (F1={worst_label[1]['f1']:.4f}), "
        f"throughput={efficiency['samples_per_second']:.2f} samples/s"
    )

    return {
        "per_label": per_label_metrics,   #type: ignore
        "efficiency": efficiency,
    }

@torch.no_grad()
def evaluate_model(param_config: ModelConfig) -> Dict[str, Any]:
    current_config = param_config
    checkpoint_path = os.path.join(current_config.checkpoint_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint for {current_config.task_name}_{current_config.unique_id_for_dir}: {checkpoint_path}")

    train_loader, val_loader, test_loader, _ = prepare_dataloaders(task_config=current_config)
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Validation loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"

    model = LegalModel(current_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(current_config.device)))

    trainer = LegalModelTrainer(model)
    trainer._remove_teacher_weight_for_evaluation()  # Ensure teacher weight is set to 0 for evaluation
    metrics = trainer.evaluate(test_loader)
    verify_metrics(metrics)
    analysis = analyze_single_task_performance_and_efficiency(
        trainer,
        test_loader,
        current_config.task_name,
        current_config.num_labels,
    )

    print(f"Evaluation metrics for {current_config.task_name}_{current_config.unique_id_for_dir}: {metrics}")
    return {"metrics": metrics, "analysis": analysis}


@torch.no_grad()
def evaluate_multi_task_model(param_model: MultiTaskModelConfig) -> Dict[str, Any]:
    current_model = param_model
    checkpoint_path = os.path.join("./datasets_store/checkpoints", current_model.unique_id_for_dir, "best_multi_task_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint for multi-task model {current_model.unique_id_for_dir}: {checkpoint_path}"
        )

    train_loaders, val_loaders, test_loaders = prepare_multitask_dataloaders(
        current_model.ledgar_config,
        current_model.unfair_tos_config,
    )
    assert len(train_loaders["ledgar"]) > 0, "LEDGAR train loader should not be empty"
    assert len(train_loaders["unfair_tos"]) > 0, "UNFAIR-ToS train loader should not be empty"
    assert len(val_loaders["ledgar"]) > 0, "LEDGAR validation loader should not be empty"
    assert len(val_loaders["unfair_tos"]) > 0, "UNFAIR-ToS validation loader should not be empty"
    assert len(test_loaders["ledgar"]) > 0, "LEDGAR test loader should not be empty"
    assert len(test_loaders["unfair_tos"]) > 0, "UNFAIR-ToS test loader should not be empty"

    mt_config = MultiTaskModelConfig(current_model.ledgar_config, current_model.unfair_tos_config, unique_id_for_dir="check_f1_multi")
    model = MultiTaskModel(mt_config)
    trainer = MultiTaskTrainer(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(trainer.device)))
    trainer._remove_teacher_weight_for_evaluation()
    metrics = trainer.evaluate(test_loaders)
    
    verify_metrics(
        metrics,
        required_keys=(
            "loss",
            "macro_f1",
            "micro_f1",
            "ledgar_loss",
            "ledgar_macro_f1",
            "ledgar_micro_f1",
            "unfair_tos_loss",
            "unfair_tos_macro_f1",
            "unfair_tos_micro_f1",
        ),
    )

    ledgar_analysis = analyze_per_label_performance_and_efficiency(
        trainer, test_loaders["ledgar"], "ledgar", current_model.ledgar_config.num_labels,
    )
    unfair_tos_analysis = analyze_per_label_performance_and_efficiency(
        trainer, test_loaders["unfair_tos"], "unfair_tos", current_model.unfair_tos_config.num_labels,
    )
    print(
        f"Evaluation metrics for multi-task model {current_model.unique_id_for_dir}: {metrics}\n"
        f"LEDGAR analysis: {ledgar_analysis['efficiency']}\n"
        f"UNFAIR-ToS analysis: {unfair_tos_analysis['efficiency']}"
    )
    return {
        "metrics": metrics,
        "analyses": {
            "ledgar": ledgar_analysis,
            "unfair_tos": unfair_tos_analysis,
        }
    }
    
def check_all_f1_scores() -> None:
    single_results = []
    multi_results = []

    for config in single_task_models_to_run:
        res = evaluate_model(param_config=config)
        single_results.append((f"{config.task_name}_{config.unique_id_for_dir}", res))

    from multi_task.train_multi_task import models_to_run as multi_task_models_to_run

    for model in multi_task_models_to_run:
        res = evaluate_multi_task_model(param_model=model)
        multi_results.append((f"{model.unique_id_for_dir}", res))

    print_table(single_results, multi_results)

if __name__ == "__main__":
    check_all_f1_scores()
