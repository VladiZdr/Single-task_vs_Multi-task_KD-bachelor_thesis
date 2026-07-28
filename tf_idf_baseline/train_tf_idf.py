import os
import logging
from typing import cast
from datasets import DatasetDict
import torch 
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from configs.model_configs import TfidfBaselineConfig
from datasets_manipulation.prepare_datasets import prep_dataset_from_raw
from tf_idf_baseline.tf_idf_model import TfidfModel
from tf_idf_baseline.tf_idf_trainer import TfidfTrainer
from single_task.train_legal_model import set_all_seeds, seed_worker
from typing import cast
from scipy.sparse import csr_matrix
import configs.model_templates as templates

logger = logging.getLogger("TfidfPipeline")

class DictTensorDataset(Dataset):
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self.x = x_tensor
        self.y = y_tensor
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx: int):
        return {"input_ids": self.x[idx], "labels": self.y[idx]}

def load_and_cache_tfidf_dataset(config: TfidfBaselineConfig) -> str:
    """
    Fits TfidfVectorizer EXCLUSIVELY on training texts to prevent data leakage.
    Transforms train, val, and test splits into sparse matrices, converts to PyTorch FloatTensors,
    and caches the dictionary payload under config.preprocessed_data_dir.
    """
    cache_path = os.path.join(config.preprocessed_data_dir, "tfidf_tensors.pt")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached TF-IDF dataset from {cache_path}")
        return cache_path

    logger.info(f"Extracting raw texts for task {config.task_name}...")
    raw_training_ds, _ = prep_dataset_from_raw(config)
    raw_training_ds = cast(DatasetDict, raw_training_ds)

    train_texts = raw_training_ds["train"]["text"]
    val_texts = raw_training_ds["validation"]["text"]
    test_texts = raw_training_ds["test"]["text"]

    train_labels = torch.tensor(raw_training_ds["train"]["labels"])
    val_labels = torch.tensor(raw_training_ds["validation"]["labels"])
    test_labels = torch.tensor(raw_training_ds["test"]["labels"])

    logger.info(f"Fitting TfidfVectorizer (max_features={config.max_features}) on train split...")
    vectorizer = TfidfVectorizer(max_features=config.max_features)
    
    X_train_sparse = cast(csr_matrix, vectorizer.fit_transform(train_texts))
    X_val_sparse = cast(csr_matrix,vectorizer.transform(val_texts))
    X_test_sparse = cast(csr_matrix,vectorizer.transform(test_texts))

    assert X_train_sparse.shape is not None
    feature_dim = X_train_sparse.shape[1]

    X_train = torch.from_numpy(X_train_sparse.toarray()).float()
    X_val = torch.from_numpy(X_val_sparse.toarray()).float()
    X_test = torch.from_numpy(X_test_sparse.toarray()).float()

    os.makedirs(config.preprocessed_data_dir, exist_ok=True)
    payload = {
        "train": (X_train, train_labels),
        "val": (X_val, val_labels),
        "test": (X_test, test_labels),
        "feature_dim": feature_dim,
    }
    torch.save(payload, cache_path)
    logger.info(f"Cached TF-IDF tensors successfully to {cache_path} with feature_dim={feature_dim}")
    return cache_path

def prepare_dataloaders(config: TfidfBaselineConfig) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
    set_all_seeds(config.seed)
    cache_path = load_and_cache_tfidf_dataset(config)
    data = torch.load(cache_path)

    if isinstance(data, dict):
        train_x, train_y = data["train"]
        val_x, val_y = data["val"]
        test_x, test_y = data["test"]
        feature_dim = int(data.get("feature_dim", train_x.shape[1]))
    else:
        train_x, train_y = data["train"]
        val_x, val_y = data["val"]
        test_x, test_y = data["test"]
        feature_dim = int(train_x.shape[1])

    train_ds = DictTensorDataset(train_x, train_y)
    val_ds = DictTensorDataset(val_x, val_y)
    test_ds = DictTensorDataset(test_x, test_y)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    unshuffled_train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, unshuffled_train_loader, feature_dim

def run_task_pipeline(config: TfidfBaselineConfig) -> None:
    logger.info(f"Running TF-IDF pipeline for {config.task_name}_{config.unique_id_for_dir}")
    train_loader, val_loader, test_loader, _, feature_dim = prepare_dataloaders(config)

    model = TfidfModel(config, input_dim=feature_dim)
    trainer = TfidfTrainer(model)

    best_weights_path = trainer.fit(train_loader, val_loader)
    if best_weights_path and os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=torch.device(config.device)))
        metrics = trainer.evaluate(test_loader)
        logger.info(f"Final Test Evaluation Metrics for {config.task_name}: {metrics}")

testers = [
    templates.tfidf_tester,
    templates.tfidf_hidden_dim_tester
]

main_models = []

models_to_run = testers

def run_pipelines():
    for model in models_to_run:
        run_task_pipeline(model)

if __name__ == "__main__":
    run_pipelines()
