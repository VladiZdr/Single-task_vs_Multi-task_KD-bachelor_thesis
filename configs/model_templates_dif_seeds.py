"""Student retraining configurations for two additional random seeds.

Teachers are intentionally imported from their original configurations.  Only
student checkpoints and outputs are seed-specific so teacher predictions can be
reused across retraining runs.
"""

from dataclasses import replace

from configs.model_configs import ModelConfig, MultiTaskModelConfig
from configs.model_templates import (
    ledgar_kd_student,
    ledgar_kd_student_annealing,
    ledgar_supervised_student_baseline,
    unfair_tos_kd_student,
    unfair_tos_kd_student_annealing,
    unfair_tos_supervised_student_baseline,
)
from configs.model_templates_low_ress import (
    ledgar_kd_student_lr4,
    ledgar_kd_student_lr6,
    ledgar_kd_student_lr8,
    ledgar_kd_student_lr10,
    ledgar_supervised_student_lr4,
    ledgar_supervised_student_lr6,
    ledgar_supervised_student_lr8,
    ledgar_supervised_student_lr10,
    unfair_tos_kd_student_lr40,
    unfair_tos_kd_student_lr60,
    unfair_tos_kd_student_lr80,
    unfair_tos_supervised_student_lr40,
    unfair_tos_supervised_student_lr60,
    unfair_tos_supervised_student_lr80,
)


def _student_with_seed(config: ModelConfig, seed: int, suffix: str) -> ModelConfig:
    """Clone a student config while isolating its run artifacts by seed."""
    return replace(
        config,
        seed=seed,
        checkpoint_dir=f"{config.checkpoint_dir}_{suffix}",
        output_dir=f"{config.output_dir}_{suffix}",
        unique_id_for_dir=f"{config.unique_id_for_dir}_{suffix}",
    )


# S2 (seed 123)
ledgar_supervised_student_baseline_S2 = _student_with_seed(ledgar_supervised_student_baseline, 123, "S2")
unfair_tos_supervised_student_baseline_S2 = _student_with_seed(unfair_tos_supervised_student_baseline, 123, "S2")
ledgar_kd_student_S2 = _student_with_seed(ledgar_kd_student, 123, "S2")
unfair_tos_kd_student_S2 = _student_with_seed(unfair_tos_kd_student, 123, "S2")
ledgar_kd_student_annealing_S2 = _student_with_seed(ledgar_kd_student_annealing, 123, "S2")
unfair_tos_kd_student_annealing_S2 = _student_with_seed(unfair_tos_kd_student_annealing, 123, "S2")

multi_task_supervised_model_S2 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_baseline_S2,
    unfair_tos_config=unfair_tos_supervised_student_baseline_S2,
    unique_id_for_dir="multi_task_model_supervised_S2",
)
multi_task_kd_model_S2 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_S2,
    unfair_tos_config=unfair_tos_kd_student_S2,
    unique_id_for_dir="multi_task_model_kd_S2",
)
multi_task_kd_model_annealing_S2 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_annealing_S2,
    unfair_tos_config=unfair_tos_kd_student_annealing_S2,
    unique_id_for_dir="multi_task_kd_annealing_S2",
)

unfair_tos_supervised_student_lr40_S2 = _student_with_seed(unfair_tos_supervised_student_lr40, 123, "S2")
ledgar_supervised_student_lr4_S2 = _student_with_seed(ledgar_supervised_student_lr4, 123, "S2")
unfair_tos_kd_student_lr40_S2 = _student_with_seed(unfair_tos_kd_student_lr40, 123, "S2")
ledgar_kd_student_lr4_S2 = _student_with_seed(ledgar_kd_student_lr4, 123, "S2")
multi_task_supervised_lr40_4_S2 = MultiTaskModelConfig(ledgar_supervised_student_lr4_S2, unfair_tos_supervised_student_lr40_S2, "MT_Supervised_LR40_4_S2")
multi_task_kd_lr40_4_S2 = MultiTaskModelConfig(ledgar_kd_student_lr4_S2, unfair_tos_kd_student_lr40_S2, "MT_KD_LR40_4_S2")

unfair_tos_supervised_student_lr60_S2 = _student_with_seed(unfair_tos_supervised_student_lr60, 123, "S2")
ledgar_supervised_student_lr6_S2 = _student_with_seed(ledgar_supervised_student_lr6, 123, "S2")
unfair_tos_kd_student_lr60_S2 = _student_with_seed(unfair_tos_kd_student_lr60, 123, "S2")
ledgar_kd_student_lr6_S2 = _student_with_seed(ledgar_kd_student_lr6, 123, "S2")
multi_task_supervised_lr60_6_S2 = MultiTaskModelConfig(ledgar_supervised_student_lr6_S2, unfair_tos_supervised_student_lr60_S2, "MT_Supervised_LR60_6_S2")
multi_task_kd_lr60_6_S2 = MultiTaskModelConfig(ledgar_kd_student_lr6_S2, unfair_tos_kd_student_lr60_S2, "MT_KD_LR60_6_S2")

unfair_tos_supervised_student_lr80_S2 = _student_with_seed(unfair_tos_supervised_student_lr80, 123, "S2")
ledgar_supervised_student_lr8_S2 = _student_with_seed(ledgar_supervised_student_lr8, 123, "S2")
unfair_tos_kd_student_lr80_S2 = _student_with_seed(unfair_tos_kd_student_lr80, 123, "S2")
ledgar_kd_student_lr8_S2 = _student_with_seed(ledgar_kd_student_lr8, 123, "S2")
multi_task_supervised_lr80_8_S2 = MultiTaskModelConfig(ledgar_supervised_student_lr8_S2, unfair_tos_supervised_student_lr80_S2, "MT_Supervised_LR80_8_S2")
multi_task_kd_lr80_8_S2 = MultiTaskModelConfig(ledgar_kd_student_lr8_S2, unfair_tos_kd_student_lr80_S2, "MT_KD_LR80_8_S2")

ledgar_supervised_student_lr10_S2 = _student_with_seed(ledgar_supervised_student_lr10, 123, "S2")
ledgar_kd_student_lr10_S2 = _student_with_seed(ledgar_kd_student_lr10, 123, "S2")
multi_task_supervised_model_lr100_10_S2 = MultiTaskModelConfig(ledgar_supervised_student_lr10_S2, unfair_tos_supervised_student_baseline_S2, "multi_task_super_lr100_10_S2")
multi_task_kd_model_lr100_10_S2 = MultiTaskModelConfig(ledgar_kd_student_lr10_S2, unfair_tos_kd_student_S2, "multi_task_kd_lr100_10_S2")


# S3 (seed 456)
ledgar_supervised_student_baseline_S3 = _student_with_seed(ledgar_supervised_student_baseline, 456, "S3")
unfair_tos_supervised_student_baseline_S3 = _student_with_seed(unfair_tos_supervised_student_baseline, 456, "S3")
ledgar_kd_student_S3 = _student_with_seed(ledgar_kd_student, 456, "S3")
unfair_tos_kd_student_S3 = _student_with_seed(unfair_tos_kd_student, 456, "S3")
ledgar_kd_student_annealing_S3 = _student_with_seed(ledgar_kd_student_annealing, 456, "S3")
unfair_tos_kd_student_annealing_S3 = _student_with_seed(unfair_tos_kd_student_annealing, 456, "S3")

multi_task_supervised_model_S3 = MultiTaskModelConfig(
    ledgar_config=ledgar_supervised_student_baseline_S3,
    unfair_tos_config=unfair_tos_supervised_student_baseline_S3,
    unique_id_for_dir="multi_task_model_supervised_S3",
)
multi_task_kd_model_S3 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_S3,
    unfair_tos_config=unfair_tos_kd_student_S3,
    unique_id_for_dir="multi_task_model_kd_S3",
)
multi_task_kd_model_annealing_S3 = MultiTaskModelConfig(
    ledgar_config=ledgar_kd_student_annealing_S3,
    unfair_tos_config=unfair_tos_kd_student_annealing_S3,
    unique_id_for_dir="multi_task_kd_annealing_S3",
)

unfair_tos_supervised_student_lr40_S3 = _student_with_seed(unfair_tos_supervised_student_lr40, 456, "S3")
ledgar_supervised_student_lr4_S3 = _student_with_seed(ledgar_supervised_student_lr4, 456, "S3")
unfair_tos_kd_student_lr40_S3 = _student_with_seed(unfair_tos_kd_student_lr40, 456, "S3")
ledgar_kd_student_lr4_S3 = _student_with_seed(ledgar_kd_student_lr4, 456, "S3")
multi_task_supervised_lr40_4_S3 = MultiTaskModelConfig(ledgar_supervised_student_lr4_S3, unfair_tos_supervised_student_lr40_S3, "MT_Supervised_LR40_4_S3")
multi_task_kd_lr40_4_S3 = MultiTaskModelConfig(ledgar_kd_student_lr4_S3, unfair_tos_kd_student_lr40_S3, "MT_KD_LR40_4_S3")

unfair_tos_supervised_student_lr60_S3 = _student_with_seed(unfair_tos_supervised_student_lr60, 456, "S3")
ledgar_supervised_student_lr6_S3 = _student_with_seed(ledgar_supervised_student_lr6, 456, "S3")
unfair_tos_kd_student_lr60_S3 = _student_with_seed(unfair_tos_kd_student_lr60, 456, "S3")
ledgar_kd_student_lr6_S3 = _student_with_seed(ledgar_kd_student_lr6, 456, "S3")
multi_task_supervised_lr60_6_S3 = MultiTaskModelConfig(ledgar_supervised_student_lr6_S3, unfair_tos_supervised_student_lr60_S3, "MT_Supervised_LR60_6_S3")
multi_task_kd_lr60_6_S3 = MultiTaskModelConfig(ledgar_kd_student_lr6_S3, unfair_tos_kd_student_lr60_S3, "MT_KD_LR60_6_S3")

unfair_tos_supervised_student_lr80_S3 = _student_with_seed(unfair_tos_supervised_student_lr80, 456, "S3")
ledgar_supervised_student_lr8_S3 = _student_with_seed(ledgar_supervised_student_lr8, 456, "S3")
unfair_tos_kd_student_lr80_S3 = _student_with_seed(unfair_tos_kd_student_lr80, 456, "S3")
ledgar_kd_student_lr8_S3 = _student_with_seed(ledgar_kd_student_lr8, 456, "S3")
multi_task_supervised_lr80_8_S3 = MultiTaskModelConfig(ledgar_supervised_student_lr8_S3, unfair_tos_supervised_student_lr80_S3, "MT_Supervised_LR80_8_S3")
multi_task_kd_lr80_8_S3 = MultiTaskModelConfig(ledgar_kd_student_lr8_S3, unfair_tos_kd_student_lr80_S3, "MT_KD_LR80_8_S3")

ledgar_supervised_student_lr10_S3 = _student_with_seed(ledgar_supervised_student_lr10, 456, "S3")
ledgar_kd_student_lr10_S3 = _student_with_seed(ledgar_kd_student_lr10, 456, "S3")
multi_task_supervised_model_lr100_10_S3 = MultiTaskModelConfig(ledgar_supervised_student_lr10_S3, unfair_tos_supervised_student_baseline_S3, "multi_task_super_lr100_10_S3")
multi_task_kd_model_lr100_10_S3 = MultiTaskModelConfig(ledgar_kd_student_lr10_S3, unfair_tos_kd_student_S3, "multi_task_kd_lr100_10_S3")


seed_2_single_task_models = [
    ledgar_supervised_student_baseline_S2, unfair_tos_supervised_student_baseline_S2,
    ledgar_kd_student_S2, unfair_tos_kd_student_S2,
    ledgar_kd_student_annealing_S2, unfair_tos_kd_student_annealing_S2,
    unfair_tos_supervised_student_lr40_S2, ledgar_supervised_student_lr4_S2,
    unfair_tos_kd_student_lr40_S2, ledgar_kd_student_lr4_S2,
    unfair_tos_supervised_student_lr60_S2, ledgar_supervised_student_lr6_S2,
    unfair_tos_kd_student_lr60_S2, ledgar_kd_student_lr6_S2,
    unfair_tos_supervised_student_lr80_S2, ledgar_supervised_student_lr8_S2,
    unfair_tos_kd_student_lr80_S2, ledgar_kd_student_lr8_S2,
    ledgar_supervised_student_lr10_S2, ledgar_kd_student_lr10_S2,
]

seed_2_multi_task_models = [
    multi_task_supervised_model_S2, multi_task_kd_model_S2, multi_task_kd_model_annealing_S2,
    multi_task_supervised_lr40_4_S2, multi_task_kd_lr40_4_S2,
    multi_task_supervised_lr60_6_S2, multi_task_kd_lr60_6_S2,
    multi_task_supervised_lr80_8_S2, multi_task_kd_lr80_8_S2,
    multi_task_supervised_model_lr100_10_S2, multi_task_kd_model_lr100_10_S2,
]

seed_3_single_task_models = [
    ledgar_supervised_student_baseline_S3, unfair_tos_supervised_student_baseline_S3,
    ledgar_kd_student_S3, unfair_tos_kd_student_S3,
    ledgar_kd_student_annealing_S3, unfair_tos_kd_student_annealing_S3,
    unfair_tos_supervised_student_lr40_S3, ledgar_supervised_student_lr4_S3,
    unfair_tos_kd_student_lr40_S3, ledgar_kd_student_lr4_S3,
    unfair_tos_supervised_student_lr60_S3, ledgar_supervised_student_lr6_S3,
    unfair_tos_kd_student_lr60_S3, ledgar_kd_student_lr6_S3,
    unfair_tos_supervised_student_lr80_S3, ledgar_supervised_student_lr8_S3,
    #unfair_tos_kd_student_lr80_S3, ledgar_kd_student_lr8_S3,
    #ledgar_supervised_student_lr10_S3, ledgar_kd_student_lr10_S3,
]

seed_3_multi_task_models = [
    multi_task_supervised_model_S3, multi_task_kd_model_S3, multi_task_kd_model_annealing_S3,
    multi_task_supervised_lr40_4_S3, multi_task_kd_lr40_4_S3,
    multi_task_supervised_lr60_6_S3, multi_task_kd_lr60_6_S3,
    multi_task_supervised_lr80_8_S3, multi_task_kd_lr80_8_S3,
    multi_task_supervised_model_lr100_10_S3, multi_task_kd_model_lr100_10_S3,
]

different_seed_single_task_models = seed_2_single_task_models + seed_3_single_task_models
different_seed_multi_task_models = seed_2_multi_task_models + seed_3_multi_task_models
