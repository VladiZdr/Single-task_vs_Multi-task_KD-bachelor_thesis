import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import model_config
from fine_tuning.export_teacher_outputs import SoftTargetExporter


def _verification_targets() -> list[tuple[str, str]]:
    return [
        ("UNFAIR-ToS teacher", model_config.unfair_tos_teacher_tester.output_dir),
        ("UNFAIR-ToS supervised student", model_config.unfair_tos_supervised_student_tester.output_dir),
        ("UNFAIR-ToS KD student", model_config.unfair_tos_kd_student_tester.output_dir),
        #("LEDGAR teacher", "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs"),
        #("LEDGAR supervised student", "./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_outputs"),
        #("LEDGAR KD student", "./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_outputs"),
    ]


def main() -> None:
    for label, directory in _verification_targets():
        print(f"\n[{label}]")
        summary = SoftTargetExporter.verify_exports(directory_path=directory)
        assert set(summary) == {"train", "validation", "test"}, f"{label} did not contain all three splits"
        for split_name, split_summary in summary.items():
            assert split_summary["num_samples"] > 0, f"{label} split '{split_name}' is empty"                           #type: ignore
            assert split_summary["num_classes"] in {8, 100}, f"{label} split '{split_name}' has unexpected class count"

    print("\nAll teacher-output exports passed verification.")


if __name__ == "__main__":
    main()
