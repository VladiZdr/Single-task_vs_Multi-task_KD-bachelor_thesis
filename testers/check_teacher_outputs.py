import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fine_tuning.export_teacher_outputs import SoftTargetExporter
from fine_tuning.train_legal_model import models_to_run


def main() -> None:
    for model in models_to_run:
        print(f"\n[{model.task_name}]")
        summary = SoftTargetExporter.verify_exports(directory_path=model.output_dir)
        assert set(summary) == {"train", "validation", "test"}, f"{model.task_name} did not contain all three splits"
        for split_name, split_summary in summary.items():
            assert split_summary["num_samples"] > 0, f"{model.task_name} split '{split_name}' is empty"                           #type: ignore
            assert split_summary["num_classes"] in {8, 100}, f"{model.task_name} split '{split_name}' has unexpected class count"

    print("\nAll teacher-output exports passed verification.")


if __name__ == "__main__":
    main()
