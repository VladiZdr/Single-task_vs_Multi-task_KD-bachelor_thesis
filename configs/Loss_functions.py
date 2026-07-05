import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunctions:
    @staticmethod    
    def get_loss_function(problem_type: str, loss_type: str, loss_reduction: str = "mean", T: float = 1.0, alpha: float = 0.5) -> nn.Module:
        if problem_type == "single_label" and loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(reduction=loss_reduction)
        elif problem_type == "multi_label" and loss_type == "bce_with_logits":
            return nn.BCEWithLogitsLoss(reduction=loss_reduction)
        elif loss_type == "kldiv":
            return KDLoss(problem_type=problem_type, T=T, alpha=alpha, reduction=loss_reduction)
        else:
            raise ValueError( f"Unsupported loss configuration: {problem_type} with loss type {loss_type}")
            

class KDLoss(nn.Module):
    def __init__(self, problem_type: str, T=1.0, alpha=0.5, reduction='mean'):
        if problem_type not in ['single_label', 'multi_label']:
            raise ValueError(f"Unsupported problem_type: {problem_type} with loss type \"kldiv\"")
        
        super(KDLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.reduction = reduction
        self.problem_type = problem_type
        self.teacher_weight = 1.0  # This will be updated during training

    def set_teacher_weight(self, weight):
        #Method to update the annealing factor externally.
        self.teacher_weight = weight

    def forward(self, student_logits, teacher_logits, labels):
        student_loss = 0.0
        distillation_loss = 0.0

        # Map generic 'mean' to 'batchmean' exclusively for KL Div to avoid warnings
        kl_reduction = 'batchmean' if self.reduction == 'mean' else self.reduction
        
        if self.problem_type == 'single_label':
            student_loss = F.cross_entropy(student_logits, labels, reduction=self.reduction)
            
            # To force developers to write safe, stable code, PyTorch's core team decided that F.kl_div will not compute the logarithm for us. 
            # It demands that we pass the student already transformed by F.log_softmax, which applies the highly stable log-sum-exp trick under the hood.
            student_soft = F.log_softmax(student_logits / self.T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)
            distillation_loss = F.kl_div(student_soft, teacher_soft, reduction=kl_reduction) * (self.T ** 2)
            
        elif self.problem_type == 'multi_label':
            labels = labels.float()  # Ensure labels are float for BCEWithLogits
            student_loss = F.binary_cross_entropy_with_logits(student_logits, labels, reduction=self.reduction)
            
            # Use standard BCE since we manually apply sigmoid to both teacher and student
            student_probs = torch.sigmoid(student_logits / self.T)
            teacher_probs = torch.sigmoid(teacher_logits / self.T)
            distillation_loss = F.binary_cross_entropy(student_probs, teacher_probs, reduction=self.reduction) * (self.T ** 2)
            
        dynamic_alpha = self.alpha * self.teacher_weight
        return (1.0 - dynamic_alpha) * student_loss + (dynamic_alpha * distillation_loss)