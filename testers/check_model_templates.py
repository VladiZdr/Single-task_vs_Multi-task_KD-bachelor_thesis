from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TemplateRecord:
    group_name: str
    index: int
    kind: str
    task_name: str
    unique_id_for_dir: str
    checkpoint_dir: str
    output_dir: str


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _iter_template_records() -> Iterable[TemplateRecord]:
    from configs.model_configs import ModelConfig, MultiTaskModelConfig, TfidfBaselineConfig
    import configs.model_templates as templates
    import configs.model_templates_testers as testers

    groups: dict[str, list[object]] = {
        "single_task_main_modules": templates.single_task_main_modules, #type: ignore
        "single_task_low_ressource_models": templates.single_task_low_ressource_models,#type: ignore
        "single_task_different_seed_models": templates.single_task_different_seed_models,#type: ignore
        "multi_task_main_modules": templates.multi_task_main_modules,#type: ignore
        "multi_task_low_ressource_models": templates.multi_task_low_ressource_models,#type: ignore
        "multi_task_different_seed_models": templates.multi_task_different_seed_models,#type: ignore
        "tf_idf_main_modules": templates.tf_idf_main_modules,#type: ignore
        "single_task_testers": testers.single_task_testers,#type: ignore
        "multi_task_testers": testers.multi_task_testers,#type: ignore
        "tf_idf_testers": testers.tf_idf_testers,#type: ignore
    }

    for group_name, configs in groups.items():
        for index, config in enumerate(configs):
            if isinstance(config, TfidfBaselineConfig):
                kind = "tfidf"
                checkpoint_dir = config.checkpoint_dir
                output_dir = config.output_dir
            elif isinstance(config, MultiTaskModelConfig):
                kind = "multi_task"
                checkpoint_dir = f"./datasets_store/checkpoints/{config.unique_id_for_dir}"
                output_dir = ""
            elif isinstance(config, ModelConfig):
                kind = "single_task"
                checkpoint_dir = config.checkpoint_dir
                output_dir = config.output_dir
            else:
                raise TypeError(f"Unsupported template type: {type(config)!r}")

            task_name = getattr(config, "task_name", "multi_task")
            unique_id_for_dir = getattr(config, "unique_id_for_dir", "")

            yield TemplateRecord(
                group_name=group_name,
                index=index,
                kind=kind,
                task_name=task_name,
                unique_id_for_dir=unique_id_for_dir,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
            )


def run_check() -> int:
    try:
        records = list(_iter_template_records())
    except ModuleNotFoundError as exc:
        print(f"Template validation could not run because a dependency is missing: {exc}")
        return 2

    print(f"Loaded {len(records)} template objects.")

    path_owners: dict[str, list[str]] = defaultdict(list)
    label_owners: dict[str, list[str]] = defaultdict(list)

    for record in records:
        owner = f"{record.group_name}[{record.index}]::{record.task_name}::{record.unique_id_for_dir}"
        label_owners[record.unique_id_for_dir].append(owner)

        if record.checkpoint_dir:
            path_owners[_normalize_path(record.checkpoint_dir)].append(f"{owner}::checkpoint")
        if record.output_dir:
            path_owners[_normalize_path(record.output_dir)].append(f"{owner}::output")

    collisions = {path: owners for path, owners in path_owners.items() if len(owners) > 1}
    reused_labels = {label: owners for label, owners in label_owners.items() if len(owners) > 1}

    print("\nPath collision scan:")
    if collisions:
        for path, owners in sorted(collisions.items()):
            print(f"  COLLISION: {path}")
            for owner in owners:
                print(f"    - {owner}")
    else:
        print("  No checkpoint/output path collisions found.")

    print("\nExperiment naming scan:")
    if reused_labels:
        for label, owners in sorted(reused_labels.items(), key=lambda item: (-len(item[1]), item[0])):
            kinds = sorted({owner.split("::")[0] for owner in owners})
            print(f"  Shared label '{label}' used by {len(owners)} configs across {', '.join(kinds)}:")
            for owner in owners:
                print(f"    - {owner}")
    else:
        print("  No reused experiment labels found.")

    print("\nTemplate check summary:")
    print("  - All templates imported successfully.")
    print("  - No direct path collisions were detected." if not collisions else "  - Path collisions must be fixed before running experiments.")
    print("  - Shared labels are reported above for naming clarity.")

    return 1 if collisions else 0


if __name__ == "__main__":
    raise SystemExit(run_check())
