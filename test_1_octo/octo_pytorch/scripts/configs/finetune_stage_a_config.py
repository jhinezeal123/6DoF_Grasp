from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert mode in ["full", "head_only", "head_mlp_only"]
    assert task in ["language_conditioned", "image_conditioned", "multimodal"]

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    else:
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )

    max_steps = FieldReference(1000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_format="torch",
        pretrained_step=None,
        batch_size=64,
        shuffle_buffer_size=1024,
        num_steps=max_steps,
        log_interval=20,
        eval_interval=1000,
        save_interval=1000,
        # If True, checkpoints will include AdamW optimizer state (much larger).
        # Keep False for inference-only checkpoints and smaller disk usage.
        save_optimizer_state=False,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(project="octo_stage_a_finetune", group=placeholder(str), entity=placeholder(str)),
        dataset_kwargs=dict(
            name="myarm_stage_a_dataset",
            data_dir=placeholder(str),
            image_obs_keys={"primary": "image_primary", "wrist": None},
            proprio_obs_key="proprio",
            language_key="language_instruction",
            action_proprio_normalization_type="normal",
            # Keep this as a supported override type (str) so it can be set from the CLI.
            # The loader treats ""/"none" as not provided.
            dataset_statistics="",
            action_normalization_mask=[True, True, True, True, True, True, False],
            standardize_fn=ModuleSpec.create(
                "octo.data.oxe.oxe_standardization_transforms:myarm_stage_a_dataset_transform",
            ),
        ),
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=200,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=128,
            num_val_batches=4,
        ),
        viz_kwargs=dict(
            eval_batch_size=32,
            trajs_for_metrics=8,
            trajs_for_viz=4,
            samples_per_state=2,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    else:
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5

    config["traj_transform_kwargs"] = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(keep_image_prob=keep_image_prob),
    )
    config["frame_transform_kwargs"] = dict(
        resize_size={"primary": (256, 256), "wrist": (128, 128)},
        image_augment_kwargs=dict(
            primary={},
            wrist={},
        ),
    )
    config["frame_transform_threads"] = 8
    return ConfigDict(config)
