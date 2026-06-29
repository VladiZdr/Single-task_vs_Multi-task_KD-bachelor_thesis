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
        elif problem_type == "single_label" and loss_type == "kldiv":
            return KDLoss(problem_type="single_label", T=T, alpha=alpha, reduction=loss_reduction)
        elif problem_type == "multi_label" and loss_type == "kldiv":
            return KDLoss(problem_type="multi_label", T=T, alpha=alpha, reduction=loss_reduction)
        else:
            raise ValueError(
                f"Unsupported loss configuration: {problem_type} "
                f"with loss type {loss_type}"
            )
            

class KDLoss(nn.Module):
    def __init__(self, problem_type: str, T=1.0, alpha=0.5, reduction='batchmean'):
        if problem_type not in ['single_label', 'multi_label']:
            raise ValueError(f"Unsupported problem_type: {self.problem_type}")
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

        # 1. Student Loss (Gold Labels)
        if self.problem_type == 'single_label':
            student_loss = F.cross_entropy(student_logits, labels)
            
            # Distillation using KL Divergence on Softmax
            student_soft = F.log_softmax(student_logits / self.T, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
            distillation_loss = F.kl_div(student_soft, teacher_soft, reduction=self.reduction) * (self.T ** 2)
            
        elif self.problem_type == 'multi_label':
            student_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
            
            # Distillation using Sigmoid for independent probabilities
            student_soft = F.logsigmoid(student_logits / self.T)
            teacher_soft = torch.sigmoid(teacher_logits / self.T)
            # Binary KL Divergence or BCE can be used here; using BCE for stability
            distillation_loss = F.binary_cross_entropy(torch.sigmoid(student_logits / self.T), 
                                                        torch.sigmoid(teacher_logits / self.T), 
                                                        reduction='mean') * (self.T ** 2)
        
        return (1 - self.alpha) * student_loss + (self.alpha * self.teacher_weight * distillation_loss)