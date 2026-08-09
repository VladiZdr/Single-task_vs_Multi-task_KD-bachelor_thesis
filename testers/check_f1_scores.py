import os
import sys
import time
from math import isfinite
from typing import Any, Dict, Iterable, Optional
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from configs.model_configs import ModelConfig, MultiTaskModelConfig, TfidfBaselineConfig
from single_task.legal_model import LegalModel
from single_task.legal_model_trainer import LegalModelTrainer
from single_task.train_legal_model import prepare_dataloaders, models_to_run as single_task_models_to_run
from multi_task.multi_task_model import MultiTaskModel
from multi_task.multi_task_trainer import MultiTaskTrainer
from multi_task.train_multi_task import prepare_multitask_dataloaders
from tf_idf_baseline.tf_idf_model import TfidfModel
from tf_idf_baseline.tf_idf_trainer import TfidfTrainer
from tf_idf_baseline.train_tf_idf import prepare_dataloaders as prepare_tfidf_dataloaders
from tf_idf_baseline.train_tf_idf import models_to_run as tfidf_models_to_run


def get_best_epoch(checkpoint_dir: str) -> str:
    """Retrieves the best epoch from the saved epoch file."""
    epoch_path = os.path.join(checkpoint_dir, "best_epoch.txt")
    if os.path.exists(epoch_path):
        with open(epoch_path, "r") as f:
            return f.read().strip()
    return "-"

def instantiate_model_and_trainer(config: ModelConfig):
    if isinstance(config, TfidfBaselineConfig):
        model = TfidfModel(config)
        trainer = TfidfTrainer(model)
    else:
        model = LegalModel(config)
        trainer = LegalModelTrainer(model)
    return model, trainer

def _format_label_str(per_label_dict: Dict[str, Dict[str, float]], find_max: bool) -> str:
    if not per_label_dict:
        return "-"
    if find_max:
        label, data = max(per_label_dict.items(), key=lambda item: item[1]["f1"])
    else:
        label, data = min(per_label_dict.items(), key=lambda item: item[1]["f1"])
    return f"{label} ({data['f1']:.2f})"

def _format_and_print_row(name: str, loss: float, macro_f1: float, micro_f1: float, best_lbl: str, worst_lbl: str, throughput: str, epoch: str = "-"):
    """Helper to print a single standard row in the summary table."""
    print(
        f"{name:<50} | "
        f"{epoch:<5} | "
        f"{loss:<7.4f} | "
        f"{macro_f1:<8.4f} | "
        f"{micro_f1:<8.4f} | "
        f"{best_lbl:<18} | "
        f"{worst_lbl:<18} | "
        f"{throughput:<10}"
    )

def _print_results_section(results, prefix: str = "", is_multitask: bool = False):
    """Prints formatted result rows for single-task, multi-task, or TF-IDF baseline models."""
    for name, res in results:
        m = res.get("metrics", {})
        epoch = res.get("epoch", "-")
        display_name = f"{prefix}{name}"

        if not is_multitask:
            # --- Single-Task / TF-IDF logic ---
            a = res.get("analysis", {})
            per_label = a.get("per_label", {})
            eff = a.get("efficiency", {})

            best_lbl = _format_label_str(per_label, find_max=True)
            worst_lbl = _format_label_str(per_label, find_max=False)
            throughput = f"{eff.get('samples_per_second', 0.0):.1f} smp/s" if eff else "-"

            _format_and_print_row(display_name, m.get("loss", 0.0), m.get("macro_f1", 0.0), m.get("micro_f1", 0.0), best_lbl, worst_lbl, throughput, epoch)

        else:
            # --- Multi-Task overall parent row ---
            _format_and_print_row(display_name, m.get("loss", 0.0), m.get("macro_f1", 0.0), m.get("micro_f1", 0.0), best_lbl="-", worst_lbl="-", throughput="-", epoch=epoch)

            # --- Multi-Task sub-task breakdown rows ---
            analyses = res.get("analyses", {})
            tasks = [("ledgar", "LEDGAR"), ("unfair_tos", "UNFAIR-ToS")]

            for idx, (task_key, task_label) in enumerate(tasks):
                t_loss = m.get(f"{task_key}_loss", 0.0)
                t_macro = m.get(f"{task_key}_macro_f1", 0.0)
                t_micro = m.get(f"{task_key}_micro_f1", 0.0)

                task_ana = analyses.get(task_key, {})
                per_label = task_ana.get("per_label", {})
                eff = task_ana.get("efficiency", {})

                best_lbl = _format_label_str(per_label, find_max=True)
                worst_lbl = _format_label_str(per_label, find_max=False)
                throughput = f"{eff.get('samples_per_second', 0.0):.1f} smp/s" if eff else "-"

                tree_branch = "└─" if idx == len(tasks) - 1 else "├─"
                sub_name = f"  {tree_branch} {task_label}"

                _format_and_print_row(sub_name, t_loss, t_macro, t_micro, best_lbl, worst_lbl, throughput, epoch="-")

def print_table(single_results, multi_results, tfidf_results):
    # Print Final Summary Table 
    header_len = 148
    print("\n" + "=" * header_len)
    print(" FINAL EVALUATION SUMMARY ".center(header_len, "="))
    print("=" * header_len)
    print(f"{'Model / Task Sub-Split':<50} | {'Epoch':<5} | {'Loss':<7} | {'Macro-F1':<8} | {'Micro-F1':<8} | {'Best Label (F1)':<18} | {'Worst Label (F1)':<18} | {'Throughput':<10}")
    print("-" * header_len)

    # 1. Single-Task
    if single_results:
        _print_results_section(single_results)

    # 2. Multi-Task
    if multi_results:
        _print_results_section(multi_results, prefix="[MultiTask] ", is_multitask=True)

    # 3. TF-IDF
    if tfidf_results:
        _print_results_section(tfidf_results, prefix="[TF-IDF] ")

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

def _collect_predictions(
    trainer: Any, 
    dataloader: Any, 
    task_name: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Unified prediction collector for MultiTaskTrainer, LegalModelTrainer, and TfidfTrainer.
    """
    all_preds = []
    all_labels = []
    start_time = time.perf_counter()
    batch_count = 0

    for batch in dataloader:
        prepared = trainer._prepare_batch(batch)
        labels = prepared["labels"]
        assert isinstance(labels, torch.Tensor)

        # 1. Resolve task name & validate if applicable
        current_task = prepared.get("task", task_name)
        if task_name is not None and "task" in prepared:
            assert prepared["task"] == task_name, f"Expected task '{task_name}', got '{prepared['task']}'"

        # 2. Determine problem_type (MultiTask vs SingleTask/TF-IDF)
        if hasattr(trainer, "task_configs") and current_task in trainer.task_configs:
            problem_type = trainer.task_configs[current_task].problem_type
        else:
            problem_type = trainer.config.problem_type

        # 3. Dynamically collect forward arguments present in prepared batch
        forward_kwargs = {
            k: prepared[k]
            for k in ("input_ids", "attention_mask", "token_type_ids")
            if k in prepared
        }
        if current_task is not None and hasattr(trainer, "task_configs"):
            forward_kwargs["task"] = current_task

        # 4. Model Forward Call
        logits = trainer.model(**forward_kwargs)

        # 5. Extract Predictions
        if problem_type == "multi_label":
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        else:
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())
        batch_count += 1

    elapsed_seconds = time.perf_counter() - start_time

    if not all_preds:
        target = f"task '{task_name}'" if task_name else "analysis"
        raise ValueError(f"No batches were processed for {target}")

    preds_array = np.concatenate(all_preds, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    return labels_array, preds_array, batch_count, elapsed_seconds

def analyze_performance_and_efficiency(
    trainer: Any,
    dataloader: Any,
    task_name: str,
    num_labels: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Unified performance and efficiency evaluation for Single-Task, Multi-Task, 
    and TF-IDF models.
    """
    # 1. Collect predictions using the unified prediction collector
    labels_array, preds_array, batch_count, elapsed_seconds = _collect_predictions(
        trainer, dataloader, task_name=task_name
    )

    # 2. Compute Precision, Recall, F1, and Support per label
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

    # 3. Format per-label metrics dictionary
    per_label_metrics: Dict[str, Dict[str, float]] = {
        f"label_{index}": {
            "precision": float(precision[index]),  # type: ignore
            "recall": float(recall[index]),        # type: ignore
            "f1": float(f1[index]),                # type: ignore
            "support": float(support[index]),      # type: ignore
        }
        for index in range(num_labels)
    }

    # 4. Calculate efficiency metrics
    total_samples = int(labels_array.shape[0])
    efficiency = {
        "elapsed_seconds": float(elapsed_seconds),
        "num_batches": float(batch_count),
        "num_samples": float(total_samples),
        "samples_per_second": float(total_samples / elapsed_seconds) if elapsed_seconds > 0 else float("inf"),
        "milliseconds_per_sample": float((elapsed_seconds / total_samples) * 1000.0) if total_samples > 0 else 0.0,
        "milliseconds_per_batch": float((elapsed_seconds / batch_count) * 1000.0) if batch_count > 0 else 0.0,
    }

    # 5. Sanity Check Assertions
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

    # 6. Log per-label summary
    best_label = max(per_label_metrics.items(), key=lambda item: item[1]["f1"])
    worst_label = min(per_label_metrics.items(), key=lambda item: item[1]["f1"])
    print(
        f"Per-label analysis for {task_name}: "
        f"best={best_label[0]} (F1={best_label[1]['f1']:.4f}), "
        f"worst={worst_label[0]} (F1={worst_label[1]['f1']:.4f}), "
        f"throughput={efficiency['samples_per_second']:.2f} samples/s"
    )

    return {
        "per_label": per_label_metrics,
        "efficiency": efficiency,
    }

@torch.no_grad()
def evaluate_single_task_model(param_config: ModelConfig) -> Dict[str, Any]:
    current_config = param_config
    checkpoint_path = os.path.join(current_config.checkpoint_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint for {current_config.task_name}_{current_config.unique_id_for_dir}: {checkpoint_path}")

    train_loader, val_loader, test_loader, _ , _= prepare_dataloaders(task_config=current_config)
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Validation loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"

    model = LegalModel(current_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(current_config.device)))

    trainer = LegalModelTrainer(model)
    trainer._remove_teacher_weight_for_evaluation()  # Ensure teacher weight is set to 0 for evaluation
    metrics = trainer.evaluate(test_loader)
    verify_metrics(metrics)
    analysis = analyze_performance_and_efficiency(
        trainer,
        test_loader,
        current_config.task_name,
        current_config.num_labels,
    )
    
    epoch = get_best_epoch(current_config.checkpoint_dir)

    print(f"Evaluation metrics for {current_config.task_name}_{current_config.unique_id_for_dir}: {metrics}")
    return {"metrics": metrics, "analysis": analysis, "epoch": epoch}

@torch.no_grad()
def evaluate_multi_task_model(param_model: MultiTaskModelConfig) -> Dict[str, Any]:
    current_model = param_model
    checkpoint_dir = os.path.join("./datasets_store/checkpoints", current_model.unique_id_for_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "best_multi_task_model.pt")

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

    ledgar_analysis = analyze_performance_and_efficiency(
        trainer, test_loaders["ledgar"], "ledgar", current_model.ledgar_config.num_labels,
    )
    unfair_tos_analysis = analyze_performance_and_efficiency(
        trainer, test_loaders["unfair_tos"], "unfair_tos", current_model.unfair_tos_config.num_labels,
    )
    
    epoch = get_best_epoch(checkpoint_dir)
    
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
        },
        "epoch": epoch
    }

@torch.no_grad()
def evaluate_tf_idf(param_config: TfidfBaselineConfig) -> Dict[str, Any]:
    current_config = param_config
    checkpoint_path = os.path.join(current_config.checkpoint_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint for TF-IDF model {current_config.task_name}_{current_config.unique_id_for_dir}: {checkpoint_path}"
        )

    train_loader, val_loader, test_loader, _, feature_dim = prepare_tfidf_dataloaders(current_config)
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Validation loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"

    model = TfidfModel(current_config, input_dim=feature_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(current_config.device)))

    trainer = TfidfTrainer(model)
    metrics = trainer.evaluate(test_loader)
    verify_metrics(metrics)
    analysis = analyze_performance_and_efficiency(
        trainer,
        test_loader,
        current_config.task_name,
        current_config.num_labels,
    )
    
    epoch = get_best_epoch(current_config.checkpoint_dir)

    print(f"Evaluation metrics for TF-IDF model {current_config.task_name}_{current_config.unique_id_for_dir}: {metrics}")
    return {"metrics": metrics, "analysis": analysis, "epoch": epoch}
    
def check_all_f1_scores() -> None:
    single_results = []
    multi_results = []
    tfidf_results = []

    for config in single_task_models_to_run:
        res = evaluate_single_task_model(param_config=config)
        single_results.append((f"{config.task_name}_{config.unique_id_for_dir}", res))

    for config in tfidf_models_to_run:
        res = evaluate_tf_idf(param_config=config)
        tfidf_results.append((f"{config.task_name}_{config.unique_id_for_dir}", res))

    from multi_task.train_multi_task import models_to_run as multi_task_models_to_run

    for model in multi_task_models_to_run:
        res = evaluate_multi_task_model(param_model=model)
        multi_results.append((f"{model.unique_id_for_dir}", res))

    print_table(single_results, multi_results, tfidf_results)

if __name__ == "__main__":
    check_all_f1_scores()