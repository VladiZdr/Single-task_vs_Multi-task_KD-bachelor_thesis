import os
import sys
from contextlib import contextmanager

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from configs.Loss_functions import KDLoss, LossFunctions
from configs.model_config import ModelConfig
import torch.nn.functional as F


@contextmanager
def expect_raises(expected_exception: type[BaseException], match: str | None = None):
    try:
        yield
    except expected_exception as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(f"Expected error message to contain '{match}', got '{exc}'") from exc
    else:
        raise AssertionError(f"Expected {expected_exception.__name__} to be raised")


def test_get_loss_function_returns_expected_modules():
    ce = LossFunctions.get_loss_function("single_label", "cross_entropy", loss_reduction="sum")
    assert isinstance(ce, torch.nn.CrossEntropyLoss)
    assert ce.reduction == "sum"

    bce = LossFunctions.get_loss_function("multi_label", "bce_with_logits", loss_reduction="mean")
    assert isinstance(bce, torch.nn.BCEWithLogitsLoss)
    assert bce.reduction == "mean"

    kd = LossFunctions.get_loss_function("multi_label", "kldiv", loss_reduction="sum", T=2.0, alpha=0.3)
    assert isinstance(kd, KDLoss)
    assert kd.problem_type == "multi_label"
    assert kd.reduction == "sum"
    assert kd.T == 2.0
    assert kd.alpha == 0.3

    with expect_raises(ValueError, "Unsupported loss configuration"):
        LossFunctions.get_loss_function("single_label", "bce_with_logits")


def test_kdloss_initialization():
    loss_single = KDLoss(problem_type="single_label")
    assert loss_single.problem_type == "single_label"

    loss_multi = KDLoss(problem_type="multi_label")
    assert loss_multi.problem_type == "multi_label"

    with expect_raises(ValueError, "Unsupported problem_type"):
        KDLoss(problem_type="invalid_type")


def test_set_teacher_weight():
    loss_fn = KDLoss(problem_type="single_label")

    assert loss_fn.teacher_weight == 1.0
    loss_fn.set_teacher_weight(0.75)
    assert loss_fn.teacher_weight == 0.75


def test_teacher_weight_schedule_helper():
    config = ModelConfig(
        task_name="unfair_tos",
        num_labels=8,
        problem_type="multi_label",
        loss_type="kldiv",
        unique_id_for_dir="sched",
        kd_teacher_weight_schedule="linear_epoch",
        kd_teacher_weight_start=1.0,
        kd_teacher_weight_end=0.0,
    )

    assert config.get_kd_teacher_weight(0, 4) == 1.0
    assert torch.isclose(torch.tensor(config.get_kd_teacher_weight(2, 4)), torch.tensor(1.0 / 3.0))
    assert config.get_kd_teacher_weight(3, 4) == 0.0
    # ADDED BOUNDARY TEST: Check if it cleanly handles out-of-bounds steps
    assert config.get_kd_teacher_weight(5, 4) == 0.0


def test_single_label_forward_and_gradients():
    batch_size = 4
    num_classes = 10
    loss_fn = KDLoss(problem_type="single_label", T=2.0, alpha=0.5)

    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    # ADDED: Check that teacher logits never collect gradients
    teacher_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))

    loss = loss_fn(student_logits, teacher_logits, labels)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    
    loss.backward()

    assert student_logits.grad is not None
    assert student_logits.grad.shape == student_logits.shape
    assert not torch.all(student_logits.grad == 0)
    # Verify teacher is untouched by the backward pass
    assert teacher_logits.grad is None or torch.all(teacher_logits.grad == 0)


def test_multi_label_forward_and_gradients():
    batch_size = 4
    num_classes = 5
    loss_fn = KDLoss(problem_type="multi_label", T=3.0, alpha=0.3)

    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()

    loss = loss_fn(student_logits, teacher_logits, labels)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert student_logits.grad is not None
    assert student_logits.grad.shape == student_logits.shape


def test_teacher_weight_impact():
    loss_fn = KDLoss(problem_type="single_label", alpha=0.5)

    student_logits = torch.randn(2, 5)
    teacher_logits = torch.randn(2, 5)
    labels = torch.tensor([1, 3])

    # 1. Get the baseline hard loss directly from PyTorch for verification
    expected_pure_hard_loss = F.cross_entropy(student_logits, labels, reduction='mean').item()

    # 2. Run with teacher active
    loss_1 = loss_fn(student_logits, teacher_logits, labels)
    
    # 3. Disable teacher influence
    loss_fn.set_teacher_weight(0.0)
    loss_0 = loss_fn(student_logits, teacher_logits, labels)

    # ASSERTIONS
    # Check that the loss actually changed when changing weights
    assert loss_1.item() != loss_0.item(), "Loss should change when teacher weight is toggled."
    
    # Check that turning off the teacher results EXACTLY in pure cross entropy loss
    assert abs(loss_0.item() - expected_pure_hard_loss) < 1e-6, "With teacher_weight=0, loss must equal pure hard loss."

def test_teacher_weight_impact_multi_label():
    loss_fn = KDLoss(problem_type="multi_label", alpha=0.5)

    student_logits = torch.randn(2, 5)
    teacher_logits = torch.randn(2, 5)
    labels = torch.randint(0, 2, (2, 5)).float()

    expected_pure_hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels, reduction='mean').item()

    loss_1 = loss_fn(student_logits, teacher_logits, labels)
    loss_fn.set_teacher_weight(0.0)
    loss_0 = loss_fn(student_logits, teacher_logits, labels)

    assert loss_1.item() != loss_0.item(), "Multi-label loss should change when teacher weight changes."
    assert abs(loss_0.item() - expected_pure_hard_loss) < 1e-6, "With weight=0, multi-label loss must equal pure BCE loss."

def test_kdloss_with_concrete_values():
    T = 2.0
    alpha = 0.5
    loss_fn = KDLoss(problem_type="single_label", T=T, alpha=alpha, reduction="mean")

    student_logits = torch.tensor([[0.5, -0.5]])
    teacher_logits = torch.tensor([[1.0, -1.0]])
    labels = torch.tensor([0])

    student_exp = torch.exp(torch.tensor(0.5))
    other_exp = torch.exp(torch.tensor(-0.5))
    student_ce_expected = -torch.log(student_exp / (student_exp + other_exp))

    student_t_logits = student_logits / T
    teacher_t_logits = teacher_logits / T
    expected_distill = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_t_logits, dim=1),
        torch.nn.functional.softmax(teacher_t_logits, dim=1),
        reduction="batchmean",
    ) * (T ** 2)

    expected_total_loss = (1 - alpha) * student_ce_expected + (alpha * expected_distill)
    actual_loss = loss_fn(student_logits, teacher_logits, labels)

    torch.testing.assert_close(
        actual_loss,
        expected_total_loss,
        msg=f"Actual Loss ({actual_loss.item():.5f}) does not match Expected Loss ({expected_total_loss.item():.5f})",
    )


def main():
    tests = [
        test_get_loss_function_returns_expected_modules,
        test_kdloss_initialization,
        test_set_teacher_weight,
        test_teacher_weight_schedule_helper,
        test_single_label_forward_and_gradients,
        test_multi_label_forward_and_gradients,
        test_teacher_weight_impact,
        test_teacher_weight_impact_multi_label,
        test_kdloss_with_concrete_values,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")

    print("All KD loss tests passed.")


if __name__ == "__main__":
    main()
