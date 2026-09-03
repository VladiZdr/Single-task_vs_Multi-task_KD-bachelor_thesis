"""
Multi-Task Knowledge Distillation Configuration Factory.

Defines model configurations for:
  1. Multi-Task Knowledge Distillation with fixed alpha = 0.5 across T in {2, 4} and 3 seeds.
  2. Low-Resource Multi-Task Knowledge Distillation with dynamic teacher annealing (1.0 -> 0.0)
     across T in {2, 4} and 3 seeds.

Teachers are imported exclusively from baseline seed S1=42 to maintain constant soft targets.
"""

from __future__ import annotations
from typing import List

from configs.model_configs import ModelConfig, MultiTaskModelConfig
from configs.model_templates import (
    ledgar_teacher,
    unfair_tos_teacher,
)
from configs.model_templates_low_ress import (
    ledgar_teacher_lr4,
    unfair_tos_teacher_lr40,
    ledgar_teacher_lr5,
    unfair_tos_teacher_lr50,
    ledgar_teacher_lr7,
    unfair_tos_teacher_lr70,
    ledgar_teacher_lr10,
)

# Experimental Constant Mappings
SEEDS = {"S1": 42, "S2": 123, "S3": 456}
TEMPERATURES = [2.0, 4.0]


def _create_task_student_pair(
    ledgar_teacher_cfg: ModelConfig,
    unfair_teacher_cfg: ModelConfig,
    low_res_ledgar_pct: int,
    low_res_unfair_pct: int,
    epochs: int,
    temperature: float,
    seed_value: int,
    seed_key: str,
    weight_schedule: str,
    alpha_start: float,
    alpha_end: float,
    experiment_id: str,
) -> tuple[ModelConfig, ModelConfig]:
    """Constructs validated LEDGAR and UNFAIR-ToS student configurations."""
    t_tag = f"t{int(temperature)}"
    
    # Enforce strict length constraint (unique_id_for_dir <= 25 characters)
    ledgar_uid = f"L_{experiment_id}_{t_tag}_{seed_key}"[:25]
    unfair_uid = f"U_{experiment_id}_{t_tag}_{seed_key}"[:25]

    ledgar_student = ModelConfig(
        task_name="ledgar",
        num_labels=100,
        problem_type="single_label",
        loss_type="kldiv",
        model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
        teacher=ledgar_teacher_cfg,
        epochs=epochs,
        low_resource_percent=low_res_ledgar_pct,
        T=temperature,
        seed=seed_value,
        kd_teacher_weight_schedule=weight_schedule,
        kd_teacher_weight_start=alpha_start,
        kd_teacher_weight_end=alpha_end,
        unique_id_for_dir=ledgar_uid,
        checkpoint_dir=f"./datasets_store/checkpoints/ledgar_{ledgar_uid}",
        output_dir=f"./datasets_store/ds_with_teacher_outputs/ledgar_{ledgar_uid}_out",
        preprocessed_data_dir=ledgar_teacher_cfg.output_dir,
    )

    unfair_student = ModelConfig(
        task_name="unfair_tos",
        num_labels=8,
        problem_type="multi_label",
        loss_type="kldiv",
        model_name_or_path="google/bert_uncased_L-4_H-256_A-4",
        teacher=unfair_teacher_cfg,
        epochs=epochs,
        low_resource_percent=low_res_unfair_pct,
        T=temperature,
        seed=seed_value,
        kd_teacher_weight_schedule=weight_schedule,
        kd_teacher_weight_start=alpha_start,
        kd_teacher_weight_end=alpha_end,
        unique_id_for_dir=unfair_uid,
        checkpoint_dir=f"./datasets_store/checkpoints/unfair_{unfair_uid}",
        output_dir=f"./datasets_store/ds_with_teacher_outputs/unfair_{unfair_uid}_out",
        preprocessed_data_dir=unfair_teacher_cfg.output_dir,
    )

    return ledgar_student, unfair_student


# ==============================================================================================
# EXPERIMENTAL SUITE 1: MULTI-TASK FIXED ALPHA = 0.5 CONFIGURATIONS
# ==============================================================================================

multi_task_fixed_alpha_configs: List[MultiTaskModelConfig] = []
single_task_fixed_alpha_students: List[ModelConfig] = []

for temp in TEMPERATURES:
    for s_key, s_val in SEEDS.items():
        led_cfg, unf_cfg = _create_task_student_pair(
            ledgar_teacher_cfg=ledgar_teacher,
            unfair_teacher_cfg=unfair_tos_teacher,
            low_res_ledgar_pct=100,
            low_res_unfair_pct=100,
            epochs=10,
            temperature=temp,
            seed_value=s_val,
            seed_key=s_key,
            weight_schedule="constant",
            alpha_start=0.5,
            alpha_end=0.5,
            experiment_id="m05",
        )
        
        mt_uid = f"mt_mix05_t{int(temp)}_{s_key}"
        mt_bundle = MultiTaskModelConfig(
            ledgar_config=led_cfg,
            unfair_tos_config=unf_cfg,
            unique_id_for_dir=mt_uid,
        )
        
        single_task_fixed_alpha_students.extend([led_cfg, unf_cfg])
        multi_task_fixed_alpha_configs.append(mt_bundle)


# ==============================================================================================
# EXPERIMENTAL SUITE 2: LOW-RESOURCE WITH TEACHER ANNEALING (1.0 -> 0.0) CONFIGURATIONS
# ==============================================================================================

multi_task_low_res_annealing_configs: List[MultiTaskModelConfig] = []
single_task_low_res_students: List[ModelConfig] = []

# Specifications: (Tag, LEDGAR Teacher, UNFAIR-ToS Teacher, LEDGAR %, UNFAIR-ToS %, Epochs)
LOW_RES_SCHEMAS = [
    ("lr40_4", ledgar_teacher_lr4, unfair_tos_teacher_lr40, 4, 40, 25),
    ("lr50_5", ledgar_teacher_lr5, unfair_tos_teacher_lr50, 5, 50, 20),
    ("lr70_7", ledgar_teacher_lr7,  unfair_tos_teacher_lr70, 7, 70, 14)
    ("lr100_10", ledgar_teacher_lr10, unfair_tos_teacher, 10, 100, 10),
]

for tag, led_t, unf_t, lr_l, lr_u, ep in LOW_RES_SCHEMAS:
    for temp in TEMPERATURES:
        for s_key, s_val in SEEDS.items():
            led_cfg, unf_cfg = _create_task_student_pair(
                ledgar_teacher_cfg=led_t,
                unfair_teacher_cfg=unf_t,
                low_res_ledgar_pct=lr_l,
                low_res_unfair_pct=lr_u,
                epochs=ep,
                temperature=temp,
                seed_value=s_val,
                seed_key=s_key,
                weight_schedule="linear_epoch",
                alpha_start=1.0,
                alpha_end=0.0,
                experiment_id=tag,
            )
            
            mt_uid = f"mt_{tag}_t{int(temp)}_{s_key}"
            mt_bundle = MultiTaskModelConfig(
                ledgar_config=led_cfg,
                unfair_tos_config=unf_cfg,
                unique_id_for_dir=mt_uid,
            )
            
            single_task_low_res_students.extend([led_cfg, unf_cfg])
            multi_task_low_res_annealing_configs.append(mt_bundle)


# Primary Export Lists for Runner Pipelines
all_multitask_models = multi_task_fixed_alpha_configs + multi_task_low_res_annealing_configs
all_singletask_submodels = single_task_fixed_alpha_students + single_task_low_res_students