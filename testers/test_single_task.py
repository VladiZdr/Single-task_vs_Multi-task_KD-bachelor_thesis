from testers.check_f1_scores import check_all_f1_scores
from testers.check_teacher_outputs import check_all_exports

if __name__ == "__main__":
    print("Starting F1 score checks...")
    check_all_f1_scores()
    print("\nStarting teacher output export checks...")
    check_all_exports()
    print("\nAll tests completed successfully.")