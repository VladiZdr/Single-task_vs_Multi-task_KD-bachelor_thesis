from configs.model_configs import ModelConfig, MultiTaskModelConfig


# ==============================================================================================
# 30% UNFAIR-ToS / 3% LEDGAR CONFIGURATIONS
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr30 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=10,
    low_resource_percent=30,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr30",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr30_outputs",
    unique_id_for_dir="Teacher_LR30",
    preprocessed_data_dir="raw"
)

ledgar_teacher_lr3 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=10,
    low_resource_percent=3,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr3",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr3_outputs",
    unique_id_for_dir="Teacher_LR3",
    preprocessed_data_dir="raw"
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr30 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=10,
    low_resource_percent=30,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr30",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr30_outputs",
    unique_id_for_dir="Supervised_LR30",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_lr3 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=10,
    low_resource_percent=3,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr3",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr3_outputs",
    unique_id_for_dir="Supervised_LR3",
    preprocessed_data_dir="raw"
)

# 3. KD Students
unfair_tos_kd_student_lr30 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_lr30,
    epochs=10,
    low_resource_percent=30,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr30",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr30_outputs",
    unique_id_for_dir="KD_LR30",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr30_outputs"
)

ledgar_kd_student_lr3 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_lr3,
    epochs=10,
    low_resource_percent=3,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr3",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr3_outputs",
    unique_id_for_dir="KD_LR3",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr3_outputs"
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr30_3 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr3,
    unfair_tos_config=unfair_tos_supervised_student_lr30,
    unique_id_for_dir="MT_Supervised_LR30_3"
)

# 5. Multi-Task KD Model
multi_task_kd_lr30_3 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr3,
    unfair_tos_config=unfair_tos_kd_student_lr30,
    unique_id_for_dir="MT_KD_LR30_3"
)


# ==============================================================================================
# 50% UNFAIR-ToS / 5% LEDGAR CONFIGURATIONS
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr50 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=10,
    low_resource_percent=50,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr50",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr50_outputs",
    unique_id_for_dir="Teacher_LR50",
    preprocessed_data_dir="raw"
)

ledgar_teacher_lr5 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=10,
    low_resource_percent=5,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr5",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr5_outputs",
    unique_id_for_dir="Teacher_LR5",
    preprocessed_data_dir="raw"
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr50 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=10,
    low_resource_percent=50,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr50",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr50_outputs",
    unique_id_for_dir="Supervised_LR50",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_lr5 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=10,
    low_resource_percent=5,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr5",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr5_outputs",
    unique_id_for_dir="Supervised_LR5",
    preprocessed_data_dir="raw"
)

# 3. KD Students
unfair_tos_kd_student_lr50 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_lr50,
    epochs=10,
    low_resource_percent=50,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr50",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr50_outputs",
    unique_id_for_dir="KD_LR50",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr50_outputs"
)

ledgar_kd_student_lr5 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_lr5,
    epochs=10,
    low_resource_percent=5,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr5",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr5_outputs",
    unique_id_for_dir="KD_LR5",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr5_outputs"
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr50_5 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr5,
    unfair_tos_config=unfair_tos_supervised_student_lr50,
    unique_id_for_dir="MT_Supervised_LR50_5"
)

# 5. Multi-Task KD Model
multi_task_kd_lr50_5 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr5,
    unfair_tos_config=unfair_tos_kd_student_lr50,
    unique_id_for_dir="MT_KD_LR50_5"
)


# ==============================================================================================
# LOW RESOURCE SINGLE-TASK MODELS
# ==============================================================================================

low_resource_single_task_models = [

    # --- 30% UNFAIR-ToS / 3% LEDGAR ---
    unfair_tos_teacher_lr30,
    ledgar_teacher_lr3,
    unfair_tos_supervised_student_lr30,
    ledgar_supervised_student_lr3,
    unfair_tos_kd_student_lr30,
    ledgar_kd_student_lr3,

    # --- 50% UNFAIR-ToS / 5% LEDGAR ---
    unfair_tos_teacher_lr50,
    ledgar_teacher_lr5,
    unfair_tos_supervised_student_lr50,
    ledgar_supervised_student_lr5,
    unfair_tos_kd_student_lr50,
    ledgar_kd_student_lr5,
]


# ==============================================================================================
# LOW RESOURCE MULTI-TASK MODELS
# ==============================================================================================

low_resource_multi_task_models = [

    # --- 30% UNFAIR-ToS / 3% LEDGAR ---
    multi_task_supervised_lr30_3,
    multi_task_kd_lr30_3,

    # --- 50% UNFAIR-ToS / 5% LEDGAR ---
    multi_task_supervised_lr50_5,
    multi_task_kd_lr50_5,
]