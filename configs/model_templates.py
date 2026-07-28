from configs.model_configs import ModelConfig, MultiTaskModelConfig, TfidfBaselineConfig


"----------------------------------------------------------------SINGLE-TASK CONFIGURATIONS---------------------------------------------------------------------------"

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
    teacher=ledgar_teacher,

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
    teacher=unfair_tos_teacher,

    epochs = 5,

    kd_teacher_weight_schedule = "linear_epoch",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_kd_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_outputs",
    unique_id_for_dir = "Single_task_KD_Student",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"
)

# Low-resource experiment configurations
ledgar_supervised_student_low_resource = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    low_resource_percent=50,
    batch_size=16,
    epochs=5,

    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student_low_resource",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_low_resource_outputs",
    unique_id_for_dir="LowResource",
    preprocessed_data_dir="raw",
)

unfair_tos_supervised_student_low_resource = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    low_resource_percent=50,
    batch_size=4,
    epochs=5,

    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_student_low_resource",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_student_low_resource_outputs",
    unique_id_for_dir="LowResource",
    preprocessed_data_dir="raw",
)

ledgar_kd_student_low_resource = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher,

    low_resource_percent=50,
    batch_size=16,
    epochs=5,

    kd_teacher_weight_schedule="linear_epoch",

    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_low_resource",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_low_resource_outputs",
    unique_id_for_dir="LowResourceKD",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
)

unfair_tos_kd_student_low_resource = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher,

    low_resource_percent=50,
    epochs=5,

    kd_teacher_weight_schedule="linear_epoch",

    device="auto",
    seed=42,

    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_student_low_resource",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_low_resource_outputs",
    unique_id_for_dir="LowResourceKD",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs",
)

# Three-seed final experiment configurations
ledgar_supervised_student_final_seed_1 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    batch_size=16,
    epochs=5,
    seed=42,

    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student_final_seed_1",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_final_seed_1_outputs",
    unique_id_for_dir="FinalSeed1",
    preprocessed_data_dir="raw",
)

ledgar_supervised_student_final_seed_2 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    batch_size=16,
    epochs=5,
    seed=43,

    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student_final_seed_2",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_final_seed_2_outputs",
    unique_id_for_dir="FinalSeed2",
    preprocessed_data_dir="raw",
)

ledgar_supervised_student_final_seed_3 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    batch_size=16,
    epochs=5,
    seed=44,

    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student_final_seed_3",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_final_seed_3_outputs",
    unique_id_for_dir="FinalSeed3",
    preprocessed_data_dir="raw",
)
#---
unfair_tos_supervised_student_final_seed_1 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    epochs=5,
    seed=42,

    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_student_final_seed_1",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_student_final_seed_1_outputs",
    unique_id_for_dir="FinalSeed1",
    preprocessed_data_dir="raw",
)

unfair_tos_supervised_student_final_seed_2 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    epochs=5,
    seed=43,

    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_student_final_seed_2",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_student_final_seed_2_outputs",
    unique_id_for_dir="FinalSeed2",
    preprocessed_data_dir="raw",
)

unfair_tos_supervised_student_final_seed_3 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    epochs=5,
    seed=44,

    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_student_final_seed_3",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_student_final_seed_3_outputs",
    unique_id_for_dir="FinalSeed3",
    preprocessed_data_dir="raw",
)
#---
ledgar_kd_student_final_seed_1 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher,

    batch_size=16,
    epochs=5,
    seed=42,

    kd_teacher_weight_schedule="linear_epoch",

    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_final_seed_1",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_final_seed_1_outputs",
    unique_id_for_dir="FinalSeed1",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
)

ledgar_kd_student_final_seed_2 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher,

    batch_size=16,
    epochs=5,
    seed=43,

    kd_teacher_weight_schedule="linear_epoch",

    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_final_seed_2",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_final_seed_2_outputs",
    unique_id_for_dir="FinalSeed2",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
)

ledgar_kd_student_final_seed_3 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher,

    batch_size=16,
    epochs=5,
    seed=44,

    kd_teacher_weight_schedule="linear_epoch",

    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_final_seed_3",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_final_seed_3_outputs",
    unique_id_for_dir="FinalSeed3",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
)
#---
unfair_tos_kd_student_final_seed_1 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher,

    epochs=5,
    seed=42,

    kd_teacher_weight_schedule="linear_epoch",

    device="auto",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_student_final_seed_1",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_final_seed_1_outputs",
    unique_id_for_dir="FinalSeed1",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs",
)

unfair_tos_kd_student_final_seed_2 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher,

    epochs=5,
    seed=43,

    kd_teacher_weight_schedule="linear_epoch",

    device="auto",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_student_final_seed_2",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_final_seed_2_outputs",
    unique_id_for_dir="FinalSeed2",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs",
)

unfair_tos_kd_student_final_seed_3 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher,

    epochs=5,
    seed=44,

    kd_teacher_weight_schedule="linear_epoch",

    device="auto",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_student_final_seed_3",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_final_seed_3_outputs",
    unique_id_for_dir="FinalSeed3",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs",
)
#---

"----------------------------------------------------------------MULTI-TASK CONFIGURATIONS----------------------------------------------------------------------------"


# Main Multi-task Supervised Model Configuration
multi_task_supervised_model = MultiTaskModelConfig(
    ledgar_config = ledgar_supervised_student_baseline,
    unfair_tos_config = unfair_tos_supervised_student_baseline,
    unique_id_for_dir = "multi_task_model_supervised"
)

# Main Multi-task KD Model Configuration
multi_task_kd_model = MultiTaskModelConfig(
    ledgar_config = ledgar_kd_student,
    unfair_tos_config = unfair_tos_kd_student,
    unique_id_for_dir = "multi_task_model_kd"
)

multi_task_supervised_model_low_resource = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_low_resource,
    unfair_tos_config=unfair_tos_supervised_student_low_resource,
    unique_id_for_dir="multi_task_low_resource_supervised",
)

multi_task_kd_model_low_resource = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_low_resource,
    unfair_tos_config=unfair_tos_kd_student_low_resource,
    unique_id_for_dir="multi_task_low_resource_kd",
)

# Main Multi-task different SEED models
multi_task_supervised_model_final_seed_1 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_final_seed_1,
    unfair_tos_config=unfair_tos_supervised_student_final_seed_1,
    unique_id_for_dir="multi_task_final_seed_1",
)

multi_task_supervised_model_final_seed_2 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_final_seed_2,
    unfair_tos_config=unfair_tos_supervised_student_final_seed_2,
    unique_id_for_dir="multi_task_final_seed_2",
)

multi_task_supervised_model_final_seed_3 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_final_seed_3,
    unfair_tos_config=unfair_tos_supervised_student_final_seed_3,
    unique_id_for_dir="multi_task_final_seed_3",
)

multi_task_kd_model_final_seed_1 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_final_seed_1,
    unfair_tos_config=unfair_tos_kd_student_final_seed_1,
    unique_id_for_dir="multi_task_kd_final_seed_1",
)

multi_task_kd_model_final_seed_2 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_final_seed_2,
    unfair_tos_config=unfair_tos_kd_student_final_seed_2,
    unique_id_for_dir="multi_task_kd_final_seed_2",
)

multi_task_kd_model_final_seed_3 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_final_seed_3,
    unfair_tos_config=unfair_tos_kd_student_final_seed_3,
    unique_id_for_dir="multi_task_kd_final_seed_3",
)

"--------------------------------------------------------------------TF-IDF BASELINE----------------------------------------------------------------------------------"

tfidf_unfair_tos = TfidfBaselineConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",

    model_name_or_path="tfidf_baseline", # !!!! important for data processing !!!!

    epochs=5,    
    batch_size=4,    

    unique_id_for_dir = "tfidf_unfair_tos",
    preprocessed_data_dir = "raw" # irrelevant -> will be overwritten later
)

tfidf_ledgar = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    
    model_name_or_path="tfidf_baseline",

    batch_size = 16,
    epochs = 5,
    
    unique_id_for_dir = "tfidf_ledgar",
    preprocessed_data_dir = "raw"
)

"------------------------------------------------------------------LISTS OF TEST MODULES----------------------------------------------------------------------------------"

single_task_main_modules = [
    # Teachers
        ledgar_teacher,
        unfair_tos_teacher,
    # Baseline Students
        ledgar_supervised_student_baseline,
        unfair_tos_supervised_student_baseline,
    # Knowledge Distillation Students
        ledgar_kd_student,
        unfair_tos_kd_student,
    # Low-resource experiments
        ledgar_supervised_student_low_resource,
        unfair_tos_supervised_student_low_resource,
        ledgar_kd_student_low_resource,
        unfair_tos_kd_student_low_resource,
    # Three-seed final experiments
        ledgar_supervised_student_final_seed_1,
        ledgar_supervised_student_final_seed_2,
        ledgar_supervised_student_final_seed_3,
        unfair_tos_supervised_student_final_seed_1,
        unfair_tos_supervised_student_final_seed_2,
        unfair_tos_supervised_student_final_seed_3,
        ledgar_kd_student_final_seed_1,
        ledgar_kd_student_final_seed_2,
        ledgar_kd_student_final_seed_3,
        unfair_tos_kd_student_final_seed_1,
        unfair_tos_kd_student_final_seed_2,
        unfair_tos_kd_student_final_seed_3,
]

multi_task_main_modules = [

    multi_task_supervised_model,
    multi_task_kd_model,

    multi_task_supervised_model_low_resource,
    multi_task_kd_model_low_resource,

    multi_task_supervised_model_final_seed_1,
    multi_task_supervised_model_final_seed_2,
    multi_task_supervised_model_final_seed_3,

    multi_task_kd_model_final_seed_1,
    multi_task_kd_model_final_seed_2,
    multi_task_kd_model_final_seed_3,
]

tf_idf_main_modules = [
    tfidf_ledgar,
    tfidf_unfair_tos
]
