import torch
import numpy as np
from .train_utils_pt import _np2pt
from octo.utils.train_utils_common import process_text

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""
    # TODO redo
    def __init__(
        self,
        rlds_dataset,
        text_processor,
        train=True,
        copy_non_writable_arrays: bool = False,
        iterator_prefetch: int = 0,
    ):
        self._rlds_dataset = rlds_dataset
        self._text_processor = text_processor
        self._is_train = train
        self._copy_non_writable_arrays = bool(copy_non_writable_arrays)
        self._iterator_prefetch = int(iterator_prefetch or 0)

    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        # Allow passing a picklable factory (callable) instead of a DLataset.
        # This is required for DataLoader(num_workers>0, multiprocessing_context='spawn'),
        # because dlimp.DLataset is not picklable.
        rlds = self._rlds_dataset
        if callable(rlds):
            try:
                rlds = rlds(worker_id=worker_id, num_workers=num_workers)
            except TypeError:
                rlds = rlds()

        # Allow passing a picklable ModuleSpec dict for text_processor.
        text_processor = self._text_processor
        if isinstance(text_processor, dict) and set(text_processor.keys()) == {"module", "name", "args", "kwargs"}:
            from octo.utils.spec import ModuleSpec
            text_processor = ModuleSpec.instantiate(text_processor)()

        # NOTE: IterableDataset + num_workers>0 will create one iterator per worker.
        # Best-effort: shard at the TF dataset level if available to avoid duplicated work.
        sharded = False
        if num_workers > 1 and hasattr(rlds, "shard"):
            try:
                rlds = rlds.shard(num_workers, worker_id)
                sharded = True
            except Exception:
                sharded = False
        # Prefer DLataset.iterator(prefetch=...) (enables TF prefetch threads) if available.
        try:
            it = rlds.iterator(prefetch=self._iterator_prefetch) if hasattr(rlds, "iterator") else rlds.as_numpy_iterator()
        except Exception:
            it = rlds.as_numpy_iterator()

        # Fallback: shard deterministically by (index % num_workers).
        for idx, sample in enumerate(it):
            if not sharded and num_workers > 1 and (idx % num_workers) != worker_id:
                continue
            del sample["dataset_name"]
            if text_processor is None:
                # Match `process_text`: if no text processor, drop language entirely.
                if "language_instruction" in sample.get("task", {}):
                    sample["task"].pop("language_instruction")
            else:
                sample["task"]["language_instruction"] = np.array(
                    [sample["task"]["language_instruction"]]
                )
                sample = process_text(sample, text_processor)

                # remove extra dim
                sample["task"]["language_instruction"]["input_ids"] = sample["task"][
                    "language_instruction"
                ]["input_ids"][0]
                sample["task"]["language_instruction"]["attention_mask"] = sample["task"][
                    "language_instruction"
                ]["attention_mask"][0]
            
            # del sample["dataset_name"]
            sample = _np2pt_batch(sample, copy_non_writable_arrays=self._copy_non_writable_arrays)
            yield sample

    def __len__(self):
        rlds = self._rlds_dataset
        # If we were constructed with a factory (callable), build once to get stats.
        if callable(rlds):
            try:
                rlds = rlds(worker_id=0, num_workers=1)
            except TypeError:
                rlds = rlds()

        if not hasattr(rlds, "dataset_statistics"):
            # Iterable datasets don't strictly need a length.
            return 0

        lengths = np.array([stats["num_transitions"] for stats in rlds.dataset_statistics])
        if hasattr(rlds, "sample_weights"):
            lengths *= np.array(rlds.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)


def _np2pt_batch(data, device=None, *, copy_non_writable_arrays: bool = False):
    # Delegate to shared helper which prefers torch.as_tensor/from_numpy (less copying)
    # and handles uint8 image transpose.
    return _np2pt(data, device=device, copy_non_writable_arrays=copy_non_writable_arrays)