from configs.Loss_functions import KDLoss
import pytest
import torch
import torch.testing as testing

def test_kdloss_initialization():
    """Test that the class initializes correctly and catches bad arguments."""
    # Valid initializations
    loss_single = KDLoss(problem_type='single_label')
    assert loss_single.problem_type == 'single_label'
    
    loss_multi = KDLoss(problem_type='multi_label')
    assert loss_multi.problem_type == 'multi_label'
    
    # Invalid initialization
    with pytest.raises(ValueError, match="Unsupported problem_type"):
        KDLoss(problem_type='invalid_type')

def test_set_teacher_weight():
    """Test that the teacher weight can be annealed/updated."""
    loss_fn = KDLoss(problem_type='single_label')
    
    assert loss_fn.teacher_weight == 1.0  # Check default
    loss_fn.set_teacher_weight(0.75)
    assert loss_fn.teacher_weight == 0.75

def test_single_label_forward_and_gradients():
    """Test the single-label forward pass and backward gradient flow."""
    batch_size = 4
    num_classes = 10
    loss_fn = KDLoss(problem_type='single_label', T=2.0, alpha=0.5)
    
    # Create mock inputs
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    loss = loss_fn(student_logits, teacher_logits, labels)
    
    # Assertions on the output
    assert loss.dim() == 0  # Loss should be a scalar
    assert not torch.isnan(loss) # Loss should not be NaN
    assert not torch.isinf(loss) # Loss should not be infinite
    
    # Backward pass
    loss.backward()
    
    # Ensure gradients are flowing back to the student model
    assert student_logits.grad is not None
    assert not torch.all(student_logits.grad == 0)

def test_multi_label_forward_and_gradients():
    """Test the multi-label forward pass and backward gradient flow."""
    batch_size = 4
    num_classes = 5
    loss_fn = KDLoss(problem_type='multi_label', T=3.0, alpha=0.3)
    
    # Create mock inputs
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    
    # Multi-label BCE requires float labels (probabilities or 0.0/1.0)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Forward pass
    loss = loss_fn(student_logits, teacher_logits, labels)
    
    # Assertions on the output
    assert loss.dim() == 0  
    assert not torch.isnan(loss)
    
    # Backward pass
    loss.backward()
    
    # Ensure gradients are flowing back to the student model
    assert student_logits.grad is not None

def test_teacher_weight_impact():
    """Test that changing the teacher weight actually changes the computed loss."""
    loss_fn = KDLoss(problem_type='single_label', alpha=0.5)
    
    student_logits = torch.randn(2, 5)
    teacher_logits = torch.randn(2, 5)
    labels = torch.tensor([1, 3])
    
    # Loss with default teacher_weight=1.0
    loss_1 = loss_fn(student_logits, teacher_logits, labels)
    
    # Loss with teacher_weight=0.0 (turns off distillation entirely)
    loss_fn.set_teacher_weight(0.0)
    loss_0 = loss_fn(student_logits, teacher_logits, labels)
    
    # Because distillation loss is usually positive, the total loss should change
    assert loss_1.item() != loss_0.item()

def test_kdloss_with_concrete_values():
    """
    Tests KDLoss 'single_label' using concrete values to ensure mathematical correctness.
    """
    # 1. Define concrete inputs
    T = 2.0
    alpha = 0.5
    loss_fn = KDLoss(problem_type='single_label', T=T, alpha=alpha, reduction='mean')
    
    # 1 batch, 2 classes
    student_logits = torch.tensor([[0.5, -0.5]])
    teacher_logits = torch.tensor([[1.0, -1.0]])
    labels = torch.tensor([0]) # Gold label is class 0
    
    # ==========================================
    # Step A: Manually compute expected Student Loss (Cross Entropy)
    # Formula: CE = -log( exp(student_logit[gold]) / sum(exp(student_logits)) )
    # ==========================================
    s_0 = torch.exp(torch.tensor(0.5))
    s_1 = torch.exp(torch.tensor(-0.5))
    student_sum = s_0 + s_1
    expected_ce_loss = -torch.log(s_0 / student_sum) 
    # Approx: 0.31326
    
    # ==========================================
    # Step B: Manually compute expected Distillation Loss (KL Div * T^2)
    # Formula: KL = sum( P_teacher * (log(P_teacher) - log(P_student)) ) * T^2
    # ==========================================
    # 1. Apply Temperature
    student_t_logits = student_logits / T  # [[0.25, -0.25]]
    teacher_t_logits = teacher_logits / T  # [[0.5, -0.5]]
    
    # 2. Compute Probabilities
    # Teacher probs
    t_t_0 = torch.exp(torch.tensor(0.5))
    t_t_1 = torch.exp(torch.tensor(-0.5))
    teacher_t_sum = t_t_0 + t_t_1
    p_teacher_0 = t_t_0 / teacher_t_sum
    p_teacher_1 = t_t_1 / teacher_t_sum
    
    # Student probs
    s_t_0 = torch.exp(torch.tensor(0.25))
    s_t_1 = torch.exp(torch.tensor(-0.25))
    student_t_sum = s_t_0 + s_t_1
    p_student_0 = s_t_0 / student_t_sum
    p_student_1 = s_t_1 / student_t_sum
    
    # 3. KL Divergence (batchmean reduction for 1 batch is just the sum)
    kl_class_0 = p_teacher_0 * (torch.log(p_teacher_0) - torch.log(p_student_0))
    kl_class_1 = p_teacher_1 * (torch.log(p_teacher_1) - torch.log(p_student_1))
    kl_divergence = kl_class_0 + kl_class_1
    
    # 4. Multiply by T^2
    expected_distill_loss = kl_divergence * (T ** 2)
    # Approx: 0.10538
    
    # ==========================================
    # Step C: Combine for Final Expected Loss
    # Formula: (1 - alpha) * CE + alpha * Distill
    # ==========================================
    expected_total_loss = (1 - alpha) * expected_ce_loss + (alpha * 1.0 * expected_distill_loss)
    # Approx: 0.20932

    # ==========================================
    # Step D: Run the KDLoss Module
    # ==========================================
    actual_loss = loss_fn(student_logits, teacher_logits, labels)

    # ==========================================
    # Step E: Assert equality using PyTorch's testing suite
    # ==========================================
    testing.assert_close(
        actual_loss, 
        expected_total_loss, 
        msg=f"Actual Loss ({actual_loss.item():.5f}) does not match Expected Loss ({expected_total_loss.item():.5f})"
    )

if __name__ == "__main__":
    test_kdloss_initialization()
    test_set_teacher_weight()
    test_single_label_forward_and_gradients()
    test_multi_label_forward_and_gradients()
    test_teacher_weight_impact()
    test_kdloss_with_concrete_values()