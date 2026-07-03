from fine_tuning.export_teacher_outputs import SoftTargetExporter
import configs.model_config

if __name__ == "__main__":
    SoftTargetExporter.verify_exports(directory_path=configs.model_config.unfair_tos_teacher_tester.output_dir)
    SoftTargetExporter.verify_exports(directory_path=configs.model_config.unfair_tos_supervised_student_tester.output_dir)
    SoftTargetExporter.verify_exports(directory_path=configs.model_config.unfair_tos_check_correct_load_preprocessed_dataset.output_dir)