from multi_task.train_multi_task import run_multitask_pipelines
from fine_tuning.train_legal_model import run_pipelines as run_single_task_pipelines

def run_all_pipelines() -> None:
    run_single_task_pipelines()
    run_multitask_pipelines()

if __name__ == "__main__":
    run_all_pipelines()