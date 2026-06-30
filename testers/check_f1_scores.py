import os
import torch
from configs.model_config import ModelConfig, ledgar_teacher_tester, unfair_tos_teacher_tester
from fine_tuning.legal_model_trainer import LegalModelTrainer
from fine_tuning.legal_model import LegalModel
from typing import Dict
from fine_tuning.train_legal_model import prepare_dataloaders

@torch.no_grad()
def evaluate_unfair_tos_teacher_tester() -> Dict[str, float]:
    unfair_tos_teacher_config = unfair_tos_teacher_tester
    
    checkpoint_filename = "best_model.pt"
    model_path = os.path.join(unfair_tos_teacher_config.checkpoint_dir, checkpoint_filename)

    # Load the dataset from disk
    train_loader, val_loader, test_loader = prepare_dataloaders(task_config=unfair_tos_teacher_config)

    # Load best model for the evaluation
    model = LegalModel(unfair_tos_teacher_config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(unfair_tos_teacher_config.device)))

    trainer = LegalModelTrainer(model, unfair_tos_teacher_config)
    metrics = trainer.evaluate(val_loader)

    print(f"Evaluation metrics for {unfair_tos_teacher_config.task_name}: {metrics}")
    return metrics

if __name__ == "__main__":
    evaluate_unfair_tos_teacher_tester()