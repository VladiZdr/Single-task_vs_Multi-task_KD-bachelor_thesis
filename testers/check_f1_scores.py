import os
import torch
import configs.model_config as model_config
from configs.model_config import ModelConfig
from fine_tuning.legal_model_trainer import LegalModelTrainer
from fine_tuning.legal_model import LegalModel
from typing import Dict
from fine_tuning.train_legal_model import prepare_dataloaders

@torch.no_grad()
def evaluate_model(param_config: ModelConfig) -> Dict[str, float]:
    current_config = param_config

    checkpoint_filename = "best_model.pt"
    model_path = os.path.join(current_config.checkpoint_dir, checkpoint_filename)

    # Load the dataset from disk
    train_loader, val_loader, test_loader = prepare_dataloaders(task_config=current_config)

    # Load best model for the evaluation
    model = LegalModel(current_config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(current_config.device)))

    trainer = LegalModelTrainer(model, current_config)
    metrics = trainer.evaluate(val_loader)

    print(f"Evaluation metrics for {current_config.task_name}_{current_config.unique_id_for_dir}: {metrics}")
    return metrics

if __name__ == "__main__":
    evaluate_model(param_config=model_config.unfair_tos_check_correct_load_preprocessed_dataset)
    #evaluate_model(param_config=model_config.unfair_tos_supervised_student_tester)
    #evaluate_model(param_config=model_config.unfair_tos_teacher_tester)
    #evaluate_model(param_config=model_config.ledgar_teacher_tester)