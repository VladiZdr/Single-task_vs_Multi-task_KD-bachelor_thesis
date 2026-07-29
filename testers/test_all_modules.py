from __future__ import annotations

from pathlib import Path

import pytest

from testers.check_f1_scores import check_all_f1_scores
from testers.check_kd_loss_fun import main as check_all_kd_loss_functions
from testers.check_teacher_outputs import check_all_exports
from testers.test_multi_task import main as check_all_multi_task
from testers.check_model_templates import run_check as check_configs


def check_all_low_resource_tests() -> None:
    test_file = Path(__file__).with_name("check_low_ressource.py")
    exit_code = pytest.main([str(test_file)])
    if exit_code != 0:
        raise SystemExit(exit_code)


def main() -> None:
    print("\nConfigs checks...")
    check_configs()
    print("\nStarting low-resource sampling checks...")
    check_all_low_resource_tests()

    print("\nStarting KD loss function checks...")
    check_all_kd_loss_functions()
    print("\nStarting multi-task checks...")
    check_all_multi_task()
    print("\nStarting teacher output export checks...")
    check_all_exports()
    print("Starting F1 score checks...")
    check_all_f1_scores()

    print("\nAll tests completed successfully.")


if __name__ == "__main__":
    main()
