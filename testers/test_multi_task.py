from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW

from configs.Loss_functions import KDLoss
from configs.model_config import ModelConfig
from multi_task.multi_task_model import MultiTaskModel
from multi_task.multi_task_trainer import MultiTaskTrainer


@contextmanager
def expect_raises(expected_exception: type[BaseException], match: str | None = None):
    try:
        yield
    except expected_exception as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(f"Expected error message to contain '{match}', got '{exc}'") from exc
    else:
        raise AssertionError(f"Expected {expected_exception.__name__} to be raised")


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 6, dropout_prob: float | None = 0.25) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        if dropout_prob is not None:
            self.config.hidden_dropout_prob = dropout_prob
        self.last_inputs: dict[str, torch.Tensor] | None = None

    def forward(self, **kwargs):  # type: ignore[override]
        self.last_inputs = kwargs
        input_ids = kwargs["input_ids"].float()
        batch_size, seq_len = input_ids.shape[:2]
        hidden_size = self.config.hidden_size

        base = input_ids.unsqueeze(-1).expand(batch_size, seq_len, hidden_size)
        return SimpleNamespace(last_hidden_state=base)


class FakeLoader:
    def __init__(self, batches: list[dict[str, torch.Tensor | list[str] | str]]) -> None:
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class TrainableMultiTaskModel(nn.Module):
    def __init__(self, ledgar_labels: int = 100, unfair_tos_labels: int = 8, unique_id: str = "trainer_test") -> None:
        super().__init__()
        self.unique_id_for_dir = unique_id
        self.ledgar_scale = nn.Parameter(torch.tensor(1.0))
        self.unfair_tos_scale = nn.Parameter(torch.tensor(0.5))
        self.num_labels = {"ledgar": ledgar_labels, "unfair_tos": unfair_tos_labels}
        self.forward_tasks: list[str | None] = []

    def forward(self, input_ids, attention_mask, token_type_ids=None, task=None):  # type: ignore[override]
        self.forward_tasks.append(task)
        if task not in self.num_labels:
            raise ValueError(f"Unsupported task '{task}'")

        scale = self.ledgar_scale if task == "ledgar" else self.unfair_tos_scale
        batch_size = input_ids.shape[0]
        num_labels = self.num_labels[task]

        base = input_ids.float().sum(dim=1, keepdim=True) + attention_mask.float().sum(dim=1, keepdim=True)
        offsets = torch.linspace(-1.0, 1.0, num_labels, device=input_ids.device).unsqueeze(0)
        return base * scale + offsets


class SpyCriterion(nn.Module):
    def __init__(self, return_value: float = 1.0) -> None:
        super().__init__()
        self.return_value = return_value
        self.calls: list[tuple[torch.Tensor, ...]] = []

    def forward(self, *args):  # type: ignore[override]
        self.calls.append(args)
        return torch.tensor(self.return_value, device=args[0].device, requires_grad=True)


def make_config(
    task_name: str,
    *,
    loss_type: str,
    problem_type: str,
    unique_id_for_dir: str,
    epochs: int = 3,
    schedule: str = "constant",
    start: float = 1.0,
    end: float = 1.0,
    model_name_or_path: str = "google/bert_uncased_L-4_H-256_A-4",
) -> ModelConfig:
    num_labels = 100 if task_name == "ledgar" else 8
    with patch("configs.model_config.os.makedirs", lambda *args, **kwargs: None):
        return ModelConfig(
            task_name=task_name,  #type: ignore
            num_labels=num_labels,  #type: ignore
            problem_type=problem_type,  #type: ignore
            loss_type=loss_type,  #type: ignore
            model_name_or_path=model_name_or_path,  #type: ignore
            epochs=epochs,
            device="cpu",
            kd_teacher_weight_schedule=schedule,  #type: ignore
            kd_teacher_weight_start=start,
            kd_teacher_weight_end=end,
            unique_id_for_dir=unique_id_for_dir,
        )


def make_kd_configs(unique_suffix: str, *, epochs: int = 3, schedule: str = "linear_epoch", start: float = 1.0, end: float = 0.0):
    ledgar = make_config(
        "ledgar",
        loss_type="kldiv",
        problem_type="single_label",
        unique_id_for_dir=f"led_{unique_suffix}",
        epochs=epochs,
        schedule=schedule,
        start=start,
        end=end,
    )
    unfair = make_config(
        "unfair_tos",
        loss_type="kldiv",
        problem_type="multi_label",
        unique_id_for_dir=f"unf_{unique_suffix}",
        epochs=epochs,
        schedule=schedule,
        start=start,
        end=end,
    )
    return ledgar, unfair


def make_supervised_configs(unique_suffix: str):
    ledgar = make_config(
        "ledgar",
        loss_type="cross_entropy",
        problem_type="single_label",
        unique_id_for_dir=f"led_sup_{unique_suffix}",
    )
    unfair = make_config(
        "unfair_tos",
        loss_type="bce_with_logits",
        problem_type="multi_label",
        unique_id_for_dir=f"unf_sup_{unique_suffix}",
    )
    return ledgar, unfair


def make_single_label_batch(task: str = "ledgar", *, include_teacher_logits: bool = True) -> dict[str, torch.Tensor | list[str]]:
    batch: dict[str, torch.Tensor | list[str]] = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 3, dtype=torch.long),
        "labels": torch.tensor([3, 7]),
        "task": [task, task],
    }
    if include_teacher_logits:
        batch["logits"] = torch.tensor(
            [[2.0] * 100, [1.0] * 100], dtype=torch.float32
        )
    return batch


def make_multi_label_batch(task: str = "unfair_tos", *, include_teacher_logits: bool = True) -> dict[str, torch.Tensor | list[str]]:
    labels = torch.zeros(2, 8, dtype=torch.float32)
    labels[0, 1] = 1.0
    labels[1, 4] = 1.0
    batch: dict[str, torch.Tensor | list[str]] = {
        "input_ids": torch.tensor([[7, 8, 9], [1, 0, 1]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": labels,
        "task": [task, task],
    }
    if include_teacher_logits:
        batch["logits"] = torch.tensor(
            [[0.3] * 8, [-0.4] * 8], dtype=torch.float32
        )
    return batch


def build_trainer(unique_suffix: str, *, kldiv: bool = True) -> tuple[MultiTaskTrainer, TrainableMultiTaskModel, ModelConfig, ModelConfig]:
    if kldiv:
        ledgar_config, unfair_config = make_kd_configs(unique_suffix)
    else:
        ledgar_config, unfair_config = make_supervised_configs(unique_suffix)

    model = TrainableMultiTaskModel(unique_id=f"trainer_{unique_suffix}")
    with patch("multi_task.multi_task_trainer.os.makedirs", lambda *args, **kwargs: None):
        trainer = MultiTaskTrainer(model, ledgar_config, unfair_config)
    return trainer, model, ledgar_config, unfair_config


def test_multitask_model_init_builds_shared_encoder_and_heads():
    ledgar_config = make_config(
        "ledgar",
        loss_type="kldiv",
        problem_type="single_label",
        unique_id_for_dir="mt_init_led",
    )
    unfair_config = make_config(
        "unfair_tos",
        loss_type="kldiv",
        problem_type="multi_label",
        unique_id_for_dir="mt_init_unf",
    )
    fake_encoder = DummyEncoder(hidden_size=7, dropout_prob=0.4)

    with patch("multi_task.multi_task_model.AutoModel.from_pretrained", return_value=fake_encoder):
        model = MultiTaskModel(ledgar_config, unfair_config, unique_id_for_dir="mt_model_init")

    assert model.unique_id_for_dir == "mt_model_init"
    assert model.encoder is fake_encoder
    assert model.dropout.p == 0.4
    assert model.classifier_heads["ledgar"].in_features == 7
    assert model.classifier_heads["ledgar"].out_features == 100
    assert model.classifier_heads["unfair_tos"].out_features == 8
    assert torch.allclose(model.classifier_heads["ledgar"].bias, torch.zeros_like(model.classifier_heads["ledgar"].bias))  #type: ignore
    assert torch.allclose(model.classifier_heads["unfair_tos"].bias, torch.zeros_like(model.classifier_heads["unfair_tos"].bias))  #type: ignore


def test_multitask_model_init_uses_default_dropout_when_encoder_config_omits_it():
    ledgar_config = make_config(
        "ledgar",
        loss_type="kldiv",
        problem_type="single_label",
        unique_id_for_dir="mt_default_drop_led",
    )
    unfair_config = make_config(
        "unfair_tos",
        loss_type="kldiv",
        problem_type="multi_label",
        unique_id_for_dir="mt_default_drop_unf",
    )
    fake_encoder = DummyEncoder(hidden_size=5, dropout_prob=None)

    with patch("multi_task.multi_task_model.AutoModel.from_pretrained", return_value=fake_encoder):
        model = MultiTaskModel(ledgar_config, unfair_config)

    assert model.dropout.p == 0.1


def test_multitask_model_init_requires_same_encoder_checkpoint():
    ledgar_config = make_config(
        "ledgar",
        loss_type="kldiv",
        problem_type="single_label",
        unique_id_for_dir="mt_mismatch_led",
        model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    )
    unfair_config = make_config(
        "unfair_tos",
        loss_type="kldiv",
        problem_type="multi_label",
        unique_id_for_dir="mt_mismatch_unf",
        model_name_or_path="nlpaueb/legal-bert-base-uncased",
    )

    with expect_raises(ValueError, "same encoder checkpoint"):
        MultiTaskModel(ledgar_config, unfair_config)


def test_multitask_model_forward_uses_selected_head_and_optional_token_type_ids():
    ledgar_config = make_config(
        "ledgar",
        loss_type="kldiv",
        problem_type="single_label",
        unique_id_for_dir="mt_forward_led",
    )
    unfair_config = make_config(
        "unfair_tos",
        loss_type="kldiv",
        problem_type="multi_label",
        unique_id_for_dir="mt_forward_unf",
    )
    fake_encoder = DummyEncoder(hidden_size=4, dropout_prob=0.0)

    with patch("multi_task.multi_task_model.AutoModel.from_pretrained", return_value=fake_encoder):
        model = MultiTaskModel(ledgar_config, unfair_config)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.ones(2, 3, dtype=torch.long)
    token_type_ids = torch.zeros(2, 3, dtype=torch.long)

    ledgar_logits = model(input_ids, attention_mask, token_type_ids=token_type_ids, task="ledgar")
    assert ledgar_logits.shape == (2, 100)
    assert "token_type_ids" in fake_encoder.last_inputs  #type: ignore
    assert torch.equal(fake_encoder.last_inputs["token_type_ids"], token_type_ids)  #type: ignore

    unfair_logits = model(input_ids, attention_mask, task="unfair_tos")
    assert unfair_logits.shape == (2, 8)
    assert "token_type_ids" not in fake_encoder.last_inputs  #type: ignore


def test_multitask_model_forward_rejects_missing_task():
    ledgar_config = make_config(
        "ledgar",
        loss_type="kldiv",
        problem_type="single_label",
        unique_id_for_dir="mt_missing_task_led",
    )
    unfair_config = make_config(
        "unfair_tos",
        loss_type="kldiv",
        problem_type="multi_label",
        unique_id_for_dir="mt_missing_task_unf",
    )
    fake_encoder = DummyEncoder(hidden_size=4)

    with patch("multi_task.multi_task_model.AutoModel.from_pretrained", return_value=fake_encoder):
        model = MultiTaskModel(ledgar_config, unfair_config)

    with expect_raises(ValueError, "requires a task name"):
        model(torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.long), task=None)


def test_multitask_model_forward_rejects_unknown_task():
    ledgar_config = make_config(
        "ledgar",
        loss_type="kldiv",
        problem_type="single_label",
        unique_id_for_dir="mt_bad_task_led",
    )
    unfair_config = make_config(
        "unfair_tos",
        loss_type="kldiv",
        problem_type="multi_label",
        unique_id_for_dir="mt_bad_task_unf",
    )
    fake_encoder = DummyEncoder(hidden_size=4)

    with patch("multi_task.multi_task_model.AutoModel.from_pretrained", return_value=fake_encoder):
        model = MultiTaskModel(ledgar_config, unfair_config)

    with expect_raises(ValueError, "Unsupported task 'other'"):
        model(torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.long), task="other")


def test_multitask_trainer_init_sets_device_checkpoint_and_teacher_weight():
    trainer, model, ledgar_config, unfair_config = build_trainer("init_kd", kldiv=True)

    assert trainer.model is model
    assert trainer.device.type == "cpu"
    assert trainer.checkpoint_path.endswith(model.unique_id_for_dir)
    assert os.path.basename(trainer.checkpoint_path) == model.unique_id_for_dir
    assert isinstance(trainer.criterions["ledgar"], KDLoss)
    assert isinstance(trainer.criterions["unfair_tos"], KDLoss)
    assert trainer.criterions["ledgar"].teacher_weight == ledgar_config.kd_teacher_weight_start
    assert trainer.criterions["unfair_tos"].teacher_weight == ledgar_config.kd_teacher_weight_start


def test_multitask_trainer_init_supports_supervised_losses():
    trainer, _, _, _ = build_trainer("init_sup", kldiv=False)

    assert isinstance(trainer.criterions["ledgar"], nn.CrossEntropyLoss)
    assert isinstance(trainer.criterions["unfair_tos"], nn.BCEWithLogitsLoss)


def test_multitask_trainer_set_teacher_weight_ignores_criterions_without_helper():
    trainer, _, _, _ = build_trainer("teacher_helper", kldiv=True)
    trainer.criterions["unfair_tos"] = object()  # type: ignore[assignment]

    trainer._set_teacher_weight(0.25)

    assert trainer.criterions["ledgar"].teacher_weight == 0.25
    assert not hasattr(trainer.criterions["unfair_tos"], "teacher_weight")


def test_multitask_trainer_sync_teacher_weight_updates_all_criteria():
    trainer, _, _, _ = build_trainer("sync_teacher", kldiv=True)

    trainer._sync_teacher_weight(1)

    assert trainer.criterions["ledgar"].teacher_weight == 0.5
    assert trainer.criterions["unfair_tos"].teacher_weight == 0.5


def test_multitask_trainer_remove_teacher_weight_for_evaluation_zeroes_weights():
    trainer, _, _, _ = build_trainer("eval_teacher", kldiv=True)

    trainer._remove_teacher_weight_for_evaluation()

    assert trainer.criterions["ledgar"].teacher_weight == 0.0
    assert trainer.criterions["unfair_tos"].teacher_weight == 0.0


def test_multitask_trainer_task_name_from_batch_handles_lists_and_rejects_invalid_batches():
    trainer, _, _, _ = build_trainer("task_name", kldiv=True)

    assert trainer._task_name_from_batch({"task": ["ledgar", "ledgar"]}) == "ledgar"
    assert trainer._task_name_from_batch({"task": ("unfair_tos", "unfair_tos")}) == "unfair_tos"

    with expect_raises(KeyError, "must contain a 'task' column"):
        trainer._task_name_from_batch({})

    with expect_raises(ValueError, "Mixed-task batches are not supported"):
        trainer._task_name_from_batch({"task": ["ledgar", "unfair_tos"]})

    with expect_raises(TypeError, "Expected a string task label"):
        trainer._task_name_from_batch({"task": 123})

    with expect_raises(ValueError, "Unknown task 'other'"):
        trainer._task_name_from_batch({"task": "other"})


def test_multitask_trainer_prepare_batch_casts_labels_and_moves_optional_fields():
    trainer, _, _, _ = build_trainer("prepare_batch", kldiv=True)
    batch = make_single_label_batch()

    prepared = trainer._prepare_batch(batch) #type: ignore

    assert prepared["task"] == "ledgar"
    assert prepared["labels"].dtype == torch.long   #type: ignore
    assert prepared["input_ids"].device.type == "cpu"   #type: ignore
    assert prepared["attention_mask"].device.type == "cpu"  #type: ignore
    assert prepared["token_type_ids"] is not None
    assert prepared["token_type_ids"].device.type == "cpu"  #type: ignore
    assert "logits" in prepared
    assert isinstance(prepared["logits"], torch.Tensor)
    assert prepared["logits"].device.type == "cpu"


def test_multitask_trainer_prepare_batch_casts_multi_label_targets_to_float_and_allows_missing_token_types():
    trainer, _, _, _ = build_trainer("prepare_multi", kldiv=True)
    batch = make_multi_label_batch()

    prepared = trainer._prepare_batch(batch)  #type: ignore

    assert prepared["task"] == "unfair_tos"
    assert prepared["labels"].dtype == torch.float32  #type: ignore
    assert prepared["token_type_ids"] is None


def test_multitask_trainer_compute_loss_handles_non_kd_criteria():
    trainer, _, _, _ = build_trainer("compute_non_kd", kldiv=False)

    ledgar_logits = torch.randn(2, 100, requires_grad=True)
    ledgar_batch = {
        "labels": torch.tensor([3, 7]),
    }
    ledgar_loss = trainer._compute_loss("ledgar", ledgar_logits, ledgar_batch)  #type: ignore
    expected_ledgar_loss = nn.CrossEntropyLoss()(ledgar_logits, ledgar_batch["labels"])
    torch.testing.assert_close(ledgar_loss, expected_ledgar_loss)

    unfair_logits = torch.randn(2, 8, requires_grad=True)
    unfair_labels = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    unfair_batch = {
        "labels": unfair_labels,
    }
    unfair_loss = trainer._compute_loss("unfair_tos", unfair_logits, unfair_batch)  #type: ignore
    expected_unfair_loss = nn.BCEWithLogitsLoss()(unfair_logits, unfair_labels)
    torch.testing.assert_close(unfair_loss, expected_unfair_loss)


def test_multitask_trainer_compute_loss_handles_kd_and_requires_teacher_logits():
    trainer, _, _, _ = build_trainer("compute_kd", kldiv=True)
    spy = SpyCriterion(return_value=2.5)
    trainer.criterions["ledgar"] = spy  # type: ignore[assignment]

    logits = torch.randn(2, 100, requires_grad=True)
    teacher_logits = torch.randn(2, 100)
    prepared_batch = {
        "labels": torch.tensor([1, 2]),
        "logits": teacher_logits,
    }

    loss = trainer._compute_loss("ledgar", logits, prepared_batch)  #type: ignore

    assert loss.item() == 2.5
    assert len(spy.calls) == 1
    assert spy.calls[0][0] is logits
    assert spy.calls[0][1] is teacher_logits
    assert torch.equal(spy.calls[0][2], prepared_batch["labels"])

    with expect_raises(KeyError, "does not contain teacher logits"):
        trainer._compute_loss("ledgar", logits, {"labels": torch.tensor([1, 2])})


def test_multitask_trainer_train_epoch_processes_tasks_sequentially():
    trainer, model, _, _ = build_trainer("train_epoch", kldiv=True)
    train_loaders = {
        "ledgar": FakeLoader([make_single_label_batch()]),      #type: ignore
        "unfair_tos": FakeLoader([make_multi_label_batch()]),   #type: ignore
    }
    optimizer = AdamW(model.parameters(), lr=1e-3)

    class DummyScheduler:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    scheduler = DummyScheduler()

    avg_loss = trainer.train_epoch(train_loaders, optimizer, scheduler)  #type: ignore

    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0
    assert scheduler.steps == 2
    assert model.forward_tasks == ["ledgar", "unfair_tos"]


def test_multitask_trainer_train_epoch_rejects_empty_inputs():
    trainer, model, _, _ = build_trainer("train_empty", kldiv=True)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    class DummyScheduler:
        def step(self) -> None:
            raise AssertionError("Should not be called")

    with expect_raises(ValueError, "Cannot train with no dataloaders"):
        trainer.train_epoch({}, optimizer, DummyScheduler())

    with expect_raises(ValueError, "No batches were processed during multi-task training"):
        trainer.train_epoch({"ledgar": FakeLoader([])}, optimizer, DummyScheduler())  #type: ignore


def test_multitask_trainer_evaluate_returns_metrics_for_both_tasks():
    trainer, model, _, _ = build_trainer("evaluate", kldiv=True)
    dataloaders = {
        "ledgar": FakeLoader([make_single_label_batch()]),  #type: ignore
        "unfair_tos": FakeLoader([make_multi_label_batch()]),  #type: ignore
    }

    metrics = trainer.evaluate(dataloaders)  #type: ignore

    assert set(
        [
            "loss",
            "macro_f1",
            "micro_f1",
            "ledgar_loss",
            "ledgar_macro_f1",
            "ledgar_micro_f1",
            "unfair_tos_loss",
            "unfair_tos_macro_f1",
            "unfair_tos_micro_f1",
        ]
    ).issubset(metrics.keys())
    assert metrics["loss"] >= 0.0
    assert model.forward_tasks == ["ledgar", "unfair_tos"]


def test_multitask_trainer_evaluate_rejects_empty_inputs():
    trainer, model, _, _ = build_trainer("evaluate_empty", kldiv=True)

    with expect_raises(ValueError, "Cannot evaluate with no dataloaders"):
        trainer.evaluate({})

    with expect_raises(ValueError, "No evaluation batches were processed"):
        trainer.evaluate({"ledgar": FakeLoader([])})  #type: ignore


def test_multitask_trainer_fit_skips_when_epochs_zero():
    trainer, model, _, _ = build_trainer("fit_skip", kldiv=True)
    trainer.ledgar_config.epochs = 0
    trainer.unfair_tos_config.epochs = 0

    result = trainer.fit({"ledgar": FakeLoader([make_single_label_batch()])}, {"ledgar": FakeLoader([make_single_label_batch()])})  #type: ignore

    assert result is None
    assert model.forward_tasks == []


def test_multitask_trainer_fit_saves_best_checkpoint_on_improvement():
    trainer, model, _, _ = build_trainer("fit_save", kldiv=True)
    train_loaders = {
        "ledgar": FakeLoader([make_single_label_batch()]),  #type: ignore
    }
    val_loaders = {
        "ledgar": FakeLoader([make_single_label_batch()]),  #type: ignore
    }
    trainer.ledgar_config.epochs = 1
    trainer.unfair_tos_config.epochs = 1

    with patch.object(trainer, "train_epoch", return_value=0.42) as train_epoch_mock, patch.object(
        trainer, "evaluate", return_value={"loss": 0.3, "macro_f1": 0.7, "micro_f1": 0.8}
    ) as evaluate_mock, patch("multi_task.multi_task_trainer.torch.save") as save_mock:
        checkpoint_path = trainer.fit(train_loaders, val_loaders)  #type: ignore

    assert train_epoch_mock.call_count == 1
    assert evaluate_mock.call_count == 1
    assert checkpoint_path is not None
    assert checkpoint_path.endswith("best_multi_task_model.pt")
    assert save_mock.call_count == 1
    assert save_mock.call_args.args[1] == checkpoint_path


def main():
    tests = [
        test_multitask_model_init_builds_shared_encoder_and_heads,
        test_multitask_model_init_uses_default_dropout_when_encoder_config_omits_it,
        test_multitask_model_init_requires_same_encoder_checkpoint,
        test_multitask_model_forward_uses_selected_head_and_optional_token_type_ids,
        test_multitask_model_forward_rejects_missing_task,
        test_multitask_model_forward_rejects_unknown_task,
        test_multitask_trainer_init_sets_device_checkpoint_and_teacher_weight,
        test_multitask_trainer_init_supports_supervised_losses,
        test_multitask_trainer_set_teacher_weight_ignores_criterions_without_helper,
        test_multitask_trainer_sync_teacher_weight_updates_all_criteria,
        test_multitask_trainer_remove_teacher_weight_for_evaluation_zeroes_weights,
        test_multitask_trainer_task_name_from_batch_handles_lists_and_rejects_invalid_batches,
        test_multitask_trainer_prepare_batch_casts_labels_and_moves_optional_fields,
        test_multitask_trainer_prepare_batch_casts_multi_label_targets_to_float_and_allows_missing_token_types,
        test_multitask_trainer_compute_loss_handles_non_kd_criteria,
        test_multitask_trainer_compute_loss_handles_kd_and_requires_teacher_logits,
        test_multitask_trainer_train_epoch_processes_tasks_sequentially,
        test_multitask_trainer_train_epoch_rejects_empty_inputs,
        test_multitask_trainer_evaluate_returns_metrics_for_both_tasks,
        test_multitask_trainer_evaluate_rejects_empty_inputs,
        test_multitask_trainer_fit_skips_when_epochs_zero,
        test_multitask_trainer_fit_saves_best_checkpoint_on_improvement,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")

    print("All multi-task class tests passed.")


if __name__ == "__main__":
    main()
