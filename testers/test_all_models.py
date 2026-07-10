from testers.check_f1_scores import check_all_f1_scores
from testers.check_teacher_outputs import check_all_exports
from testers.check_kd_loss_fun import main as check_all_kd_loss_functions
from testers.check_teacher_outputs import check_all_exports
from testers.test_multi_task import main as check_all_multi_task

if __name__ == "__main__":
    print("Starting F1 score checks...")
    check_all_f1_scores()
    print("\nStarting teacher output export checks...")
    check_all_exports()
    print("\nStarting KD loss function checks...")
    check_all_kd_loss_functions()
    print("\nStarting teacher output export checks...")
    check_all_exports()
    print("\nStarting multi-task checks...")
    check_all_multi_task()

    print("\nAll tests completed successfully.")