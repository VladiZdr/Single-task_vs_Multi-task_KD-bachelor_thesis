from configs.model_config import ModelConfig
from multi_task.multi_task_model import MultiTaskModel


"------------------------SINGLE-TASK CONFIGURATIONS------------------------"
# Testers  
ledgar_teacher_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=1,  

    batch_size = 16,

    unique_id_for_dir = "tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_teacher_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=1,  
    
    batch_size=4,    

    unique_id_for_dir = "tester",
    preprocessed_data_dir = "raw"
)

ledgar_supervised_student_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,
    batch_size = 16,

    unique_id_for_dir = "supervised_student_tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_supervised_student_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,  
    
    batch_size=4,
    
    unique_id_for_dir = "supervised_student_tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_check_correct_load_preprocessed_dataset = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=1,  
    
    batch_size=4,
    
    unique_id_for_dir = "check_correct_load",
    preprocessed_data_dir = "./datasets_store/unfair_tos_preprocessed"
)

unfair_tos_kd_student_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,  
    
    batch_size=4,

    kd_teacher_weight_schedule = "linear_epoch",

    unique_id_for_dir = "kd_student_tester",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs_tester"
)

ledgar_kd_student_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,  

    batch_size = 16,

    kd_teacher_weight_schedule = "linear_epoch",

    unique_id_for_dir = "kd_student_tester",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs_tester"
)

# Teachers
ledgar_teacher = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    batch_size = 16,
    epochs = 5,
    
    checkpoint_dir = "./datasets_store/checkpoints/ledgar_teacher",
    output_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
    unique_id_for_dir = "Teacher",
    preprocessed_data_dir = "raw"
)

unfair_tos_teacher = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    epochs = 5,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_teacher",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs",
    unique_id_for_dir = "Teacher",
    preprocessed_data_dir = "raw"
)

# Supervised Students Baselines   
ledgar_supervised_student_baseline = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    batch_size = 16,
    epochs = 5,

    checkpoint_dir = "./datasets_store/checkpoints/ledgar_supervised_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_outputs",
    unique_id_for_dir = "Baseline",
    preprocessed_data_dir = "raw"
)

unfair_tos_supervised_student_baseline = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    epochs = 5,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_supervised_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_student_outputs",
    unique_id_for_dir = "Baseline",
    preprocessed_data_dir = "raw"
)
 
# Single-task Knowledge Distillation Students
ledgar_kd_student = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    batch_size = 16,
    epochs = 5,

    kd_teacher_weight_schedule = "linear_epoch",
    
    checkpoint_dir = "./datasets_store/checkpoints/ledgar_kd_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_outputs",
    unique_id_for_dir = "Single_task_KD_Student",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs"
)

unfair_tos_kd_student = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    epochs = 5,

    kd_teacher_weight_schedule = "linear_epoch",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_kd_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_outputs",
    unique_id_for_dir = "Single_task_KD_Student",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"
)


"------------------------MULTI-TASK CONFIGURATIONS------------------------"

#Testers

multi_task_kd_model_tester = MultiTaskModel(
    ledgar_config = ledgar_kd_student_tester,
    unfair_tos_config = unfair_tos_kd_student_tester,
    unique_id_for_dir = "multi_task_model_tester"
)

multi_task_supervised_model_tester = MultiTaskModel(
    ledgar_config = ledgar_supervised_student_tester,
    unfair_tos_config = unfair_tos_supervised_student_tester,
    unique_id_for_dir = "multi_task_model_supervised_tester"
)

# Main Multi-task Supervised Model Configuration

multi_task_supervised_model = MultiTaskModel(
    ledgar_config = ledgar_supervised_student_baseline,
    unfair_tos_config = unfair_tos_supervised_student_baseline,
    unique_id_for_dir = "multi_task_model_supervised"
)

# Main Multi-task KD Model Configuration

multi_task_kd_model = MultiTaskModel(
    ledgar_config = ledgar_kd_student,
    unfair_tos_config = unfair_tos_kd_student,
    unique_id_for_dir = "multi_task_model_main"
)