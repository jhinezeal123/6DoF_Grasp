from ml_collections import ConfigDict
from octo.utils.spec import ModuleSpec


def get_config(config_string=None):
    cfg = ConfigDict()
    cfg.seed = 42
    cfg.num_steps = 50000
    cfg.log_interval = 50
    cfg.eval_interval = 1000
    cfg.save_interval = 10000
    cfg.grad_accum_steps = 1
    cfg.save_dir = "/home/ceec/khoanhd/VLAR_Lab/octo_weights/finetuned_pytorch/octo-small-1.5-torch-ft-ovft-myarm_stage_a_single_dataset"
    cfg.append_wandb_id_to_save_dir = True
    cfg.overwrite_output = False
    cfg.pretrained_path = "/home/ceec/khoanhd/VLAR_Lab/octo_weights/pretrained_pytorch/octo-small-1.5-torch"
    cfg.pretrained_format = "torch"
    cfg.pretrained_step = None
    cfg.load_optimizer_state = False
    # Finetune typically restarts the schedule at step 0 (do NOT inherit the pretrained checkpoint step).
    cfg.start_step = 0
    cfg.optimizer = ConfigDict(dict(
        learning_rate=dict(name="cosine", init_value=0.0, peak_value=3e-4, warmup_steps=2000, decay_steps=50000),
        weight_decay=0.01,
        clip_gradient=1.0,
        frozen_keys=(".*hf_model.*",),
    ))
    cfg.dataset_kwargs = ConfigDict(dict(
        batch_size=32,
        shuffle_buffer_size=4096,
        # Match the pretrained Octo checkpoint's expected trajectory shapes.
        # - window_size: observation history length (horizon)
        # - action_horizon: number of (current+future) actions per timestep
        traj_transform_kwargs=dict(
            # Add goal observations into the task dict (matches Octo pretraining).
            # This provides task images (e.g. goal image) for the visual tokenizers' task_stack_keys.
            goal_relabeling_strategy="uniform",
            window_size=2,
            action_horizon=4,
            max_action_dim=7,
            # Proprio intentionally disabled end-to-end for this run.
            max_proprio_dim=None,
        ),
        # Match the pretrained Octo checkpoint's expected image patch token counts.
        # primary: 256x256 -> 16x16 patches -> 256 tokens
        # wrist:   128x128 ->  8x8 patches ->  64 tokens
        frame_transform_kwargs=dict(
            resize_size=dict(primary=(256, 256), wrist=(128, 128)),
        ),
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        dataloader_multiprocessing_context="spawn",
        rlds_iterator_prefetch=8,
        tf_intra_op_threads=8,
        tf_inter_op_threads=8,
        # Avoid PyTorch warning + any undefined behavior risk from read-only numpy buffers.
        # (Slight CPU/memory overhead due to copies.)
        copy_non_writable_arrays=True,
    ))
    cfg.dataset_json = "scripts/datasets/myarm_stage_a_single_dataset.json"
    cfg.text_processor = ModuleSpec.create(
        "octo.data.utils.text_processing:HFTokenizer",
        tokenizer_name="t5-base",
        encode_with_model=False,
        tokenizer_kwargs={
            "max_length": 16,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
    )
    cfg.wandb = dict(project="octo-small-1.5-torch-ft-ovft-myarm_stage_a_single_dataset", group=None, entity=None)
    cfg.eval_datasets = ()
    return cfg
