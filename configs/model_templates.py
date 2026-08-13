from configs.model_configs import ModelConfig, MultiTaskModelConfig, TfidfBaselineConfig


"----------------------------------------------------------------SINGLE-TASK CONFIGURATIONS---------------------------------------------------------------------------"

# Teachers
ledgar_teacher = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    epochs = 10,
    
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

    epochs = 10,

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

    epochs = 10,

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

    epochs = 10,

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

    epochs = 10,

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

    epochs = 10,

    kd_teacher_weight_schedule = "linear_epoch",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_kd_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_outputs",
    unique_id_for_dir = "Single_task_KD_Student",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"
)

# Constant 0.5 Teacher Weight Baselines
ledgar_kd_student_mix_05 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher,
    epochs=10,
    
    kd_teacher_weight_schedule="constant",
    kd_teacher_weight_start=0.5,
    
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_mix_05",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_mix_05_outputs",
    unique_id_for_dir="Mix_05",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs"
)

unfair_tos_kd_student_mix_05 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher,
    epochs=10,
    
    kd_teacher_weight_schedule="constant",
    kd_teacher_weight_start=0.5,
    
    device="auto",
    seed=42,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_student_mix_05",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_mix_05_outputs",
    unique_id_for_dir="Mix_05",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"
)

# Constant 0.7 Teacher Weight Baselines
ledgar_kd_student_mix_07 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher,
    epochs=10,
    
    kd_teacher_weight_schedule="constant",
    kd_teacher_weight_start=0.7,
    
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_mix_07",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_mix_07_outputs",
    unique_id_for_dir="Mix_07",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs"
)

unfair_tos_kd_student_mix_07 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher,
    epochs=10,
    
    kd_teacher_weight_schedule="constant",
    kd_teacher_weight_start=0.7,
    
    device="auto",
    seed=42,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_student_mix_07",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_mix_07_outputs",
    unique_id_for_dir="Mix_07",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"
)

# Low-resource experiment configurations
ledgar_supervised_student_low_resource = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    low_resource_percent=50,
    epochs=10,

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
    epochs=10,

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
    epochs=10,

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
    epochs=10,

    kd_teacher_weight_schedule="linear_epoch",

    device="auto",
    seed=42,

    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_student_low_resource",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_low_resource_outputs",
    unique_id_for_dir="LowResourceKD",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs",
)

# Three-seed final experiment configurations

ledgar_supervised_student_final_seed_2 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    epochs=10,
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

    epochs=10,
    seed=44,

    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student_final_seed_3",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_final_seed_3_outputs",
    unique_id_for_dir="FinalSeed3",
    preprocessed_data_dir="raw",
)
#---

unfair_tos_supervised_student_final_seed_2 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    epochs=10,
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

    epochs=10,
    seed=44,

    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_student_final_seed_3",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_student_final_seed_3_outputs",
    unique_id_for_dir="FinalSeed3",
    preprocessed_data_dir="raw",
)
#---
ledgar_kd_student_final_seed_2 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher,

    epochs=10,
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

    epochs=10,
    seed=44,

    kd_teacher_weight_schedule="linear_epoch",

    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_final_seed_3",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_final_seed_3_outputs",
    unique_id_for_dir="FinalSeed3",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
)
#---


unfair_tos_kd_student_final_seed_2 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher,

    epochs=10,
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

    epochs=10,
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

multi_task_kd_model_mix_05 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_mix_05,
    unfair_tos_config=unfair_tos_kd_student_mix_05,
    unique_id_for_dir="multi_task_kd_mix_05"
)

multi_task_kd_model_mix_07 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_mix_07,
    unfair_tos_config=unfair_tos_kd_student_mix_07,
    unique_id_for_dir="multi_task_kd_mix_07"
)

# Main Multi-task different SEED models
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

    epochs=10,    

    unique_id_for_dir = "tfidf_unfair_tos",
    preprocessed_data_dir = "raw" # irrelevant -> will be overwritten later
)

tfidf_ledgar = TfidfBaselineConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    
    model_name_or_path="tfidf_baseline",

    epochs = 10,
    
    unique_id_for_dir = "tfidf_ledgar",
    preprocessed_data_dir = "raw"
)

"------------------------------------------------------------------Balanced TEST MODULES----------------------------------------------------------------------------------"

ledgar_teacher_balanced = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=10,
    
    # Downsample ~60k dataset to ~5.5k to match UNFAIR-ToS
    low_resource_percent=9,
    
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_balanced",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_balanced_outputs",
    unique_id_for_dir="Teacher_Bal",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_balanced = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=10,
    
    low_resource_percent=9,
    
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student_bal",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_bal_outputs",
    unique_id_for_dir="Baseline_Bal",
    preprocessed_data_dir="raw"
)

ledgar_kd_student_balanced = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_balanced, # Points to the balanced teacher
    epochs=10,
    
    kd_teacher_weight_schedule="linear_epoch",
    
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_bal",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_bal_outputs",
    unique_id_for_dir="KD_Student_Bal",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_balanced_outputs"
)

multi_task_supervised_model_balanced = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_balanced,
    unfair_tos_config=unfair_tos_supervised_student_baseline,
    unique_id_for_dir="multi_task_super_bal"
)

multi_task_kd_model_balanced = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_balanced,
    unfair_tos_config=unfair_tos_kd_student,
    unique_id_for_dir="multi_task_kd_bal"
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
]
constants_single_task_models = [
    ledgar_kd_student_mix_05,
    unfair_tos_kd_student_mix_05,
    ledgar_kd_student_mix_07,
    unfair_tos_kd_student_mix_07,
]
balanced_single_task_models = [
    ledgar_teacher_balanced,
    ledgar_supervised_student_balanced,
    ledgar_kd_student_balanced,
]
single_task_low_ressource_models = [
    # Low-resource experiments
    ledgar_supervised_student_low_resource,
    unfair_tos_supervised_student_low_resource,
    ledgar_kd_student_low_resource,
    unfair_tos_kd_student_low_resource,
]
single_task_different_seed_models = [
    # Three-seed final experiments
    ledgar_supervised_student_final_seed_2,
    ledgar_supervised_student_final_seed_3,
    
    unfair_tos_supervised_student_final_seed_2,
    unfair_tos_supervised_student_final_seed_3,
    
    ledgar_kd_student_final_seed_2,
    ledgar_kd_student_final_seed_3,
    
    unfair_tos_kd_student_final_seed_2,
    unfair_tos_kd_student_final_seed_3,
]

multi_task_main_modules = [
    multi_task_supervised_model,
    multi_task_kd_model,
]
constants_multi_task_models = [
    multi_task_kd_model_mix_05,
    multi_task_kd_model_mix_07,
]
balanced_multi_task_models = [
    multi_task_supervised_model_balanced,
    multi_task_kd_model_balanced,
]
multi_task_low_ressource_models =[
    multi_task_supervised_model_low_resource,
    multi_task_kd_model_low_resource,
]
multi_task_different_seed_models = [
    multi_task_supervised_model_final_seed_2,
    multi_task_supervised_model_final_seed_3,
    
    multi_task_kd_model_final_seed_2,
    multi_task_kd_model_final_seed_3,
]

tf_idf_main_modules = [
    tfidf_ledgar,
    tfidf_unfair_tos
]
