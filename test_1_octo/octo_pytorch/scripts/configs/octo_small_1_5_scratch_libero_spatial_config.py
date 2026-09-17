from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict, FieldReference

get_base_config = imp.load_source(
    "config_small_libero_scratch", os.path.join(os.path.dirname(__file__), "config_small_libero_scratch.py")
).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    # NOTE: In this repo, the base config's signature is `get_config(transformer_size="vit_s")`.
    # Older upstream variants used `config_string` and parsed it internally.
    # Passing `None` positionally would override `transformer_size` with None and crash.
    if config_string in (None, ""):
        config = get_base_config()
    else:
        # Treat config_string as transformer_size (e.g. "vit_s", "vit_b").
        config = get_base_config(config_string)

    action_dim = FieldReference(7)

    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
    }
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizer,
            encoder="t5-base",
            finetune_encoder=False,
        ),
    }
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHead,
        readout_key="readout_action",
        use_map=False,
        action_horizon=4,
        action_dim=action_dim,
        n_diffusion_samples=1,
        dropout_rate=0.0,
    )

    # Primary-only camera
    primary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
    }
    # Keep TF frame transforms light/consistent.
    config["dataset_kwargs"]["frame_transform_kwargs"]["num_parallel_calls"] = 8

    config = update_config(
        config,
        # Optimizer steps (not micro-steps)
        num_steps=50000,
        # Gradient accumulation: effective_batch ~= per_device_batch_size * world_size * grad_accum_steps
        grad_accum_steps=1,
        window_size=2,
        optimizer=dict(
            learning_rate=dict(
                name="rsqrt",
                init_value=0.0,
                peak_value=8e-5,
                warmup_steps=2000,
                timescale=10000,
            ),
            weight_decay=0.1,
            clip_gradient=1.0,
            frozen_keys=("*hf_model*",),
        ),
        log_interval=50,
        eval_interval=500,
        save_interval=10000,
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="oxe_magic_soup",
                data_dir="gs://rail-orca-central2/resize_256_256",
                load_camera_views=("primary",),
                load_depth=False,
                force_recompute_dataset_statistics=False,
            ),
            traj_transform_kwargs=dict(
                action_horizon=4,
                max_action_dim=action_dim,
                task_augment_strategy="delete_and_rephrase",
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=0.5,
                ),
            ),
            # Interpret as *per-device* batch via train_pt.py (preferred).
            per_device_batch_size=4,
            # Fallback for older scripts that only read `batch_size`.
            batch_size=4,
            shuffle_buffer_size=4096,
            balance_weights=True,
            traj_transform_threads=16,
            traj_read_threads=16,
        ),
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(
            ModuleSpec.create(
                hf_weights_loader,
                hf_model="t5-base",
            ),
        ),
        eval_datasets=["libero_spatial"],
    )


    return config
