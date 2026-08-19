from configs.model_configs import ModelConfig, MultiTaskModelConfig #[cite: 2]
from configs.model_templates import unfair_tos_supervised_student_baseline, unfair_tos_kd_student #[cite: 2]

# ==============================================================================================
# 30% UNFAIR-ToS / 3% LEDGAR CONFIGURATIONS (33 Epochs)
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr30 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=33, # Proportionally increased: 10 * (100/30)
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
    epochs=33, # Matched with UNFAIR-ToS
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
    epochs=33, 
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
    epochs=33, 
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
    epochs=33, 
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
    epochs=33, 
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
# 40% UNFAIR-ToS / 4% LEDGAR CONFIGURATIONS (25 Epochs)
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr40 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=25, # Proportionally increased: 10 * (100/40)
    low_resource_percent=40,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr40",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr40_outputs",
    unique_id_for_dir="Teacher_LR40",
    preprocessed_data_dir="raw"
)

ledgar_teacher_lr4 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=25, # Matched with UNFAIR-ToS
    low_resource_percent=4,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr4",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr4_outputs",
    unique_id_for_dir="Teacher_LR4",
    preprocessed_data_dir="raw"
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr40 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=25, 
    low_resource_percent=40,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr40",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr40_outputs",
    unique_id_for_dir="Supervised_LR40",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_lr4 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=25, 
    low_resource_percent=4,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr4",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr4_outputs",
    unique_id_for_dir="Supervised_LR4",
    preprocessed_data_dir="raw"
)

# 3. KD Students
unfair_tos_kd_student_lr40 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_lr40,
    epochs=25, 
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr40",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr40_outputs",
    unique_id_for_dir="KD_LR40",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr40_outputs"
)

ledgar_kd_student_lr4 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_lr4,
    epochs=25, 
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr4",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr4_outputs",
    unique_id_for_dir="KD_LR4",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr4_outputs"
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr40_4 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr4,
    unfair_tos_config=unfair_tos_supervised_student_lr40,
    unique_id_for_dir="MT_Supervised_LR40_4"
)

# 5. Multi-Task KD Model
multi_task_kd_lr40_4 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr4,
    unfair_tos_config=unfair_tos_kd_student_lr40,
    unique_id_for_dir="MT_KD_LR40_4"
)

# ==============================================================================================
# 50% UNFAIR-ToS / 5% LEDGAR CONFIGURATIONS (20 Epochs)
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr50 = ModelConfig(
    task_name="unfair_tos", #[cite: 2]
    num_labels=8, #[cite: 2]
    problem_type="multi_label", #[cite: 2]
    loss_type="bce_with_logits", #[cite: 2]
    model_name_or_path="nlpaueb/legal-bert-base-uncased", #[cite: 2]
    epochs=20, # Proportionally increased: 10 * (100/50)[cite: 1]
    low_resource_percent=50, #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr50", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr50_outputs", #[cite: 2]
    unique_id_for_dir="Teacher_LR50", #[cite: 2]
    preprocessed_data_dir="raw" #[cite: 2]
)

ledgar_teacher_lr5 = ModelConfig(
    task_name="ledgar", #[cite: 2]
    num_labels=100, #[cite: 2]
    problem_type="single_label", #[cite: 2]
    loss_type="cross_entropy", #[cite: 2]
    model_name_or_path="nlpaueb/legal-bert-base-uncased", #[cite: 2]
    epochs=20, # Matched with UNFAIR-ToS[cite: 1]
    low_resource_percent=5, #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr5", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr5_outputs", #[cite: 2]
    unique_id_for_dir="Teacher_LR5", #[cite: 2]
    preprocessed_data_dir="raw" #[cite: 2]
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr50 = ModelConfig(
    task_name="unfair_tos", #[cite: 2]
    num_labels=8, #[cite: 2]
    problem_type="multi_label", #[cite: 2]
    loss_type="bce_with_logits", #[cite: 2]
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4", #[cite: 2]
    epochs=20, #[cite: 1]
    low_resource_percent=50, #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr50", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr50_outputs", #[cite: 2]
    unique_id_for_dir="Supervised_LR50", #[cite: 2]
    preprocessed_data_dir="raw" #[cite: 2]
)

ledgar_supervised_student_lr5 = ModelConfig(
    task_name="ledgar", #[cite: 2]
    num_labels=100, #[cite: 2]
    problem_type="single_label", #[cite: 2]
    loss_type="cross_entropy", #[cite: 2]
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4", #[cite: 2]
    epochs=20, #[cite: 1]
    low_resource_percent=5, #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr5", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr5_outputs", #[cite: 2]
    unique_id_for_dir="Supervised_LR5", #[cite: 2]
    preprocessed_data_dir="raw" #[cite: 2]
)

# 3. KD Students
unfair_tos_kd_student_lr50 = ModelConfig(
    task_name="unfair_tos", #[cite: 2]
    num_labels=8, #[cite: 2]
    problem_type="multi_label", #[cite: 2]
    loss_type="kldiv", #[cite: 2]
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4", #[cite: 2]
    teacher=unfair_tos_teacher_lr50, #[cite: 2]
    epochs=20, #[cite: 1]
    kd_teacher_weight_schedule="linear_epoch", #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr50", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr50_outputs", #[cite: 2]
    unique_id_for_dir="KD_LR50", #[cite: 2]
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr50_outputs" #[cite: 2]
)

ledgar_kd_student_lr5 = ModelConfig(
    task_name="ledgar", #[cite: 2]
    num_labels=100, #[cite: 2]
    problem_type="single_label", #[cite: 2]
    loss_type="kldiv", #[cite: 2]
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4", #[cite: 2]
    teacher=ledgar_teacher_lr5, #[cite: 2]
    epochs=20, #[cite: 1]
    kd_teacher_weight_schedule="linear_epoch", #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr5", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr5_outputs", #[cite: 2]
    unique_id_for_dir="KD_LR5", #[cite: 2]
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr5_outputs" #[cite: 2]
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr50_5 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr5, #[cite: 2]
    unfair_tos_config=unfair_tos_supervised_student_lr50, #[cite: 2]
    unique_id_for_dir="MT_Supervised_LR50_5" #[cite: 2]
)

# 5. Multi-Task KD Model
multi_task_kd_lr50_5 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr5, #[cite: 2]
    unfair_tos_config=unfair_tos_kd_student_lr50, #[cite: 2]
    unique_id_for_dir="MT_KD_LR50_5" #[cite: 2]
)


# ==============================================================================================
# 60% UNFAIR-ToS / 6% LEDGAR CONFIGURATIONS (17 Epochs)
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr60 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=17, # Proportionally increased: 10 * (100/60)[cite: 1]
    low_resource_percent=60,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr60",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr60_outputs",
    unique_id_for_dir="Teacher_LR60",
    preprocessed_data_dir="raw"
)

ledgar_teacher_lr6 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=17, # Matched with UNFAIR-ToS[cite: 1]
    low_resource_percent=6,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr6",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr6_outputs",
    unique_id_for_dir="Teacher_LR6",
    preprocessed_data_dir="raw"
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr60 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=17,
    low_resource_percent=60,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr60",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr60_outputs",
    unique_id_for_dir="Supervised_LR60",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_lr6 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=17,
    low_resource_percent=6,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr6",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr6_outputs",
    unique_id_for_dir="Supervised_LR6",
    preprocessed_data_dir="raw"
)

# 3. KD Students
unfair_tos_kd_student_lr60 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_lr60,
    epochs=17,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr60",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr60_outputs",
    unique_id_for_dir="KD_LR60",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr60_outputs"
)

ledgar_kd_student_lr6 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_lr6,
    epochs=17,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr6",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr6_outputs",
    unique_id_for_dir="KD_LR6",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr6_outputs"
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr60_6 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr6,
    unfair_tos_config=unfair_tos_supervised_student_lr60,
    unique_id_for_dir="MT_Supervised_LR60_6"
)

# 5. Multi-Task KD Model
multi_task_kd_lr60_6 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr6,
    unfair_tos_config=unfair_tos_kd_student_lr60,
    unique_id_for_dir="MT_KD_LR60_6"
)


# ==============================================================================================
# 70% UNFAIR-ToS / 7% LEDGAR CONFIGURATIONS (14 Epochs)
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr70 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=14, # Proportionally increased: 10 * (100/70)[cite: 1]
    low_resource_percent=70,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr70",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr70_outputs",
    unique_id_for_dir="Teacher_LR70",
    preprocessed_data_dir="raw"
)

ledgar_teacher_lr7 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=14, # Matched with UNFAIR-ToS[cite: 1]
    low_resource_percent=7,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr7",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr7_outputs",
    unique_id_for_dir="Teacher_LR7",
    preprocessed_data_dir="raw"
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr70 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=14,
    low_resource_percent=70,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr70",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr70_outputs",
    unique_id_for_dir="Supervised_LR70",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_lr7 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=14,
    low_resource_percent=7,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr7",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr7_outputs",
    unique_id_for_dir="Supervised_LR7",
    preprocessed_data_dir="raw"
)

# 3. KD Students
unfair_tos_kd_student_lr70 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_lr70,
    epochs=14,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr70",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr70_outputs",
    unique_id_for_dir="KD_LR70",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr70_outputs"
)

ledgar_kd_student_lr7 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_lr7,
    epochs=14,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr7",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr7_outputs",
    unique_id_for_dir="KD_LR7",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr7_outputs"
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr70_7 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr7,
    unfair_tos_config=unfair_tos_supervised_student_lr70,
    unique_id_for_dir="MT_Supervised_LR70_7"
)

# 5. Multi-Task KD Model
multi_task_kd_lr70_7 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr7,
    unfair_tos_config=unfair_tos_kd_student_lr70,
    unique_id_for_dir="MT_KD_LR70_7"
)


# ==============================================================================================
# 80% UNFAIR-ToS / 8% LEDGAR CONFIGURATIONS (13 Epochs)
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr80 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=13, # Proportionally increased: 10 * (100/80)[cite: 1]
    low_resource_percent=80,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr80",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr80_outputs",
    unique_id_for_dir="Teacher_LR80",
    preprocessed_data_dir="raw"
)

ledgar_teacher_lr8 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=13, # Matched with UNFAIR-ToS[cite: 1]
    low_resource_percent=8,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr8",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr8_outputs",
    unique_id_for_dir="Teacher_LR8",
    preprocessed_data_dir="raw"
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr80 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=13,
    low_resource_percent=80,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr80",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr80_outputs",
    unique_id_for_dir="Supervised_LR80",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_lr8 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=13,
    low_resource_percent=8,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr8",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr8_outputs",
    unique_id_for_dir="Supervised_LR8",
    preprocessed_data_dir="raw"
)

# 3. KD Students
unfair_tos_kd_student_lr80 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_lr80,
    epochs=13,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr80",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr80_outputs",
    unique_id_for_dir="KD_LR80",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr80_outputs"
)

ledgar_kd_student_lr8 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_lr8,
    epochs=13,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr8",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr8_outputs",
    unique_id_for_dir="KD_LR8",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr8_outputs"
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr80_8 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr8,
    unfair_tos_config=unfair_tos_supervised_student_lr80,
    unique_id_for_dir="MT_Supervised_LR80_8"
)

# 5. Multi-Task KD Model
multi_task_kd_lr80_8 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr8,
    unfair_tos_config=unfair_tos_kd_student_lr80,
    unique_id_for_dir="MT_KD_LR80_8"
)


# ==============================================================================================
# 90% UNFAIR-ToS / 9% LEDGAR CONFIGURATIONS (11 Epochs)
# ==============================================================================================

# 1. Teachers
unfair_tos_teacher_lr90 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=11, # Proportionally increased: 10 * (100/90)[cite: 1]
    low_resource_percent=90,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_teacher_lr90",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr90_outputs",
    unique_id_for_dir="Teacher_LR90",
    preprocessed_data_dir="raw"
)

ledgar_teacher_lr9 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",
    epochs=11, # Matched with UNFAIR-ToS[cite: 1]
    low_resource_percent=9,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr9",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr9_outputs",
    unique_id_for_dir="Teacher_LR9",
    preprocessed_data_dir="raw"
)

# 2. Supervised Student Baselines
unfair_tos_supervised_student_lr90 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=11,
    low_resource_percent=90,
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_supervised_lr90",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_lr90_outputs",
    unique_id_for_dir="Supervised_LR90",
    preprocessed_data_dir="raw"
)

ledgar_supervised_student_lr9 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    epochs=11,
    low_resource_percent=9,
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_lr9",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_lr9_outputs",
    unique_id_for_dir="Supervised_LR9",
    preprocessed_data_dir="raw"
)

# 3. KD Students
unfair_tos_kd_student_lr90 = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=unfair_tos_teacher_lr90,
    epochs=11,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/unfair_tos_kd_lr90",
    output_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_lr90_outputs",
    unique_id_for_dir="KD_LR90",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_lr90_outputs"
)

ledgar_kd_student_lr9 = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
    teacher=ledgar_teacher_lr9,
    epochs=11,
    kd_teacher_weight_schedule="linear_epoch",
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_lr9",
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_lr9_outputs",
    unique_id_for_dir="KD_LR9",
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr9_outputs"
)

# 4. Multi-Task Supervised Model
multi_task_supervised_lr90_9 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr9,
    unfair_tos_config=unfair_tos_supervised_student_lr90,
    unique_id_for_dir="MT_Supervised_LR90_9"
)

# 5. Multi-Task KD Model
multi_task_kd_lr90_9 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr9,
    unfair_tos_config=unfair_tos_kd_student_lr90,
    unique_id_for_dir="MT_KD_LR90_9"
)


# ==============================================================================================
# 100% UNFAIR-ToS / 10% LEDGAR CONFIGURATIONS (10 Epochs)
# ==============================================================================================

ledgar_teacher_lr10 = ModelConfig(
    task_name="ledgar", #[cite: 2]
    num_labels=100, #[cite: 2]
    problem_type="single_label", #[cite: 2]
    loss_type="cross_entropy", #[cite: 2]
    model_name_or_path="nlpaueb/legal-bert-base-uncased", #[cite: 2]
    epochs=10, # Kept proportional at 10 * (100/100)[cite: 1]
    low_resource_percent=10, # Downsample ~60k dataset to ~5.5k to match UNFAIR-ToS[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/ledgar_teacher_lr10", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr10_outputs", #[cite: 2]
    unique_id_for_dir="Teacher_LR10", #[cite: 2]
    preprocessed_data_dir="raw" #[cite: 2]
)

ledgar_supervised_student_lr10 = ModelConfig(
    task_name="ledgar", #[cite: 2]
    num_labels=100, #[cite: 2]
    problem_type="single_label", #[cite: 2]
    loss_type="cross_entropy", #[cite: 2]
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4", #[cite: 2]
    epochs=10, #[cite: 1, 2]
    low_resource_percent=10, #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/ledgar_supervised_student_lr10", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_lr10_outputs", #[cite: 2]
    unique_id_for_dir="Baseline_LR10", #[cite: 2]
    preprocessed_data_dir="raw" #[cite: 2]
)

ledgar_kd_student_lr10 = ModelConfig(
    task_name="ledgar", #[cite: 2]
    num_labels=100, #[cite: 2]
    problem_type="single_label", #[cite: 2]
    loss_type="kldiv", #[cite: 2]
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4", #[cite: 2]
    teacher=ledgar_teacher_lr10, # Points to the low-resource teacher[cite: 2]
    epochs=10, #[cite: 1, 2]
    kd_teacher_weight_schedule="linear_epoch", #[cite: 2]
    checkpoint_dir="./datasets_store/checkpoints/ledgar_kd_student_lr10", #[cite: 2]
    output_dir="./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_lr10_outputs", #[cite: 2]
    unique_id_for_dir="KD_Student_LR10", #[cite: 2]
    preprocessed_data_dir="./datasets_store/ds_with_teacher_outputs/ledgar_teacher_lr10_outputs" #[cite: 2]
)

multi_task_supervised_model_lr100_10 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_lr10, #[cite: 2]
    unfair_tos_config=unfair_tos_supervised_student_baseline, #[cite: 2]
    unique_id_for_dir="multi_task_super_lr100_10" #[cite: 2]
)

multi_task_kd_model_lr100_10 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_lr10, #[cite: 2]
    unfair_tos_config=unfair_tos_kd_student, #[cite: 2]
    unique_id_for_dir="multi_task_kd_lr100_10" #[cite: 2]
)


# ==============================================================================================
# LOW RESOURCE SINGLE-TASK MODELS
# ==============================================================================================

low_resource_single_task_models = [
    # --- 30% UNFAIR-ToS / 3% LEDGAR ---
    ledgar_teacher_lr3,
    unfair_tos_teacher_lr30,
    unfair_tos_supervised_student_lr30,
    ledgar_supervised_student_lr3,
    unfair_tos_kd_student_lr30,
    ledgar_kd_student_lr3,

    # --- 40% UNFAIR-ToS / 4% LEDGAR ---
    ledgar_teacher_lr4,
    unfair_tos_teacher_lr40,
    unfair_tos_supervised_student_lr40,
    ledgar_supervised_student_lr4,
    unfair_tos_kd_student_lr40,
    ledgar_kd_student_lr4,

    # --- 50% UNFAIR-ToS / 5% LEDGAR ---
    #already trained (not evaluated!!) ledgar_teacher_lr5, #[cite: 2]
    #already trained (not evaluated!!) unfair_tos_teacher_lr50, #[cite: 2]
    #already trained (not evaluated!!) unfair_tos_supervised_student_lr50, #[cite: 2]
    #already trained (not evaluated!!) ledgar_supervised_student_lr5, #[cite: 2]
    unfair_tos_kd_student_lr50, #[cite: 2]
    ledgar_kd_student_lr5, #[cite: 2]

    # --- 60% UNFAIR-ToS / 6% LEDGAR ---
    unfair_tos_teacher_lr60,
    ledgar_teacher_lr6,
    unfair_tos_supervised_student_lr60,
    ledgar_supervised_student_lr6,
    unfair_tos_kd_student_lr60,
    ledgar_kd_student_lr6,

    # --- 70% UNFAIR-ToS / 7% LEDGAR ---
    unfair_tos_teacher_lr70,
    ledgar_teacher_lr7,
    unfair_tos_supervised_student_lr70,
    ledgar_supervised_student_lr7,
    unfair_tos_kd_student_lr70,
    ledgar_kd_student_lr7,

    # --- 80% UNFAIR-ToS / 8% LEDGAR ---
    unfair_tos_teacher_lr80,
    ledgar_teacher_lr8,
    unfair_tos_supervised_student_lr80,
    ledgar_supervised_student_lr8,
    unfair_tos_kd_student_lr80,
    ledgar_kd_student_lr8,

    # --- 90% UNFAIR-ToS / 9% LEDGAR ---
    unfair_tos_teacher_lr90,
    ledgar_teacher_lr9,
    unfair_tos_supervised_student_lr90,
    ledgar_supervised_student_lr9,
    unfair_tos_kd_student_lr90,
    ledgar_kd_student_lr9,

    # --- 100% UNFAIR-ToS / 10% LEDGAR ---
    ledgar_teacher_lr10, #[cite: 2]
    ledgar_supervised_student_lr10, #[cite: 2]
    ledgar_kd_student_lr10, #[cite: 2]
]

# ==============================================================================================
# LOW RESOURCE MULTI-TASK MODELS
# ==============================================================================================

low_resource_multi_task_models = [
    # --- 30% UNFAIR-ToS / 3% LEDGAR ---
    multi_task_supervised_lr30_3,
    multi_task_kd_lr30_3,

    # --- 40% UNFAIR-ToS / 4% LEDGAR ---
    multi_task_supervised_lr40_4,
    multi_task_kd_lr40_4,

    # --- 50% UNFAIR-ToS / 5% LEDGAR ---
    multi_task_supervised_lr50_5, #[cite: 2]
    multi_task_kd_lr50_5, #[cite: 2]

    # --- 60% UNFAIR-ToS / 6% LEDGAR ---
    multi_task_supervised_lr60_6,
    multi_task_kd_lr60_6,

    # --- 70% UNFAIR-ToS / 7% LEDGAR ---
    multi_task_supervised_lr70_7,
    multi_task_kd_lr70_7,

    # --- 80% UNFAIR-ToS / 8% LEDGAR ---
    multi_task_supervised_lr80_8,
    multi_task_kd_lr80_8,

    # --- 90% UNFAIR-ToS / 9% LEDGAR ---
    multi_task_supervised_lr90_9,
    multi_task_kd_lr90_9,

    # --- 100% UNFAIR-ToS / 10% LEDGAR ---
    multi_task_supervised_model_lr100_10, #[cite: 2]
    multi_task_kd_model_lr100_10, #[cite: 2]
]