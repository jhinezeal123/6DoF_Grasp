from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

MULTI_MODULE = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


class TextProcessor(ABC):
    """
    Base class for text tokenization or text embedding.
    """

    @abstractmethod
    def encode(self, strings: Sequence[str]):
        raise NotImplementedError


class HFTokenizer(TextProcessor):
    def __init__(
        self,
        tokenizer_name: str,
        tokenizer_kwargs: Optional[dict] = {
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
        encode_with_model: bool = False,
    ):
        # Keep this torch-only friendly: do NOT import Flax/JAX at import-time.
        # If encode_with_model=True, we use the PyTorch model backend.
        from transformers import AutoTokenizer  # lazy import

        requested_max_length = None
        if tokenizer_kwargs is not None:
            requested_max_length = tokenizer_kwargs.get("max_length")
        if requested_max_length is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Setting model_max_length avoids a noisy T5TokenizerFast FutureWarning while
            # keeping the actual truncation behavior controlled by tokenizer_kwargs.
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                model_max_length=int(requested_max_length),
            )
        self.encode_with_model = encode_with_model

        # Default kwargs depend on the backend we use.
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {
                "max_length": 64,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt" if encode_with_model else "np",
            }
        else:
            # If the caller asked to encode with a model, ensure tensors are torch tensors.
            if encode_with_model:
                tokenizer_kwargs = dict(tokenizer_kwargs)
                tokenizer_kwargs["return_tensors"] = "pt"

        self.tokenizer_kwargs = tokenizer_kwargs

        self.model = None
        if self.encode_with_model:
            from transformers import AutoModel  # PyTorch backend

            self.model = AutoModel.from_pretrained(tokenizer_name)
            self.model.eval()

    def encode(self, strings: Sequence[str]):
        # this creates another nested layer with "input_ids", "attention_mask", etc.
        inputs = self.tokenizer(
            strings,
            **self.tokenizer_kwargs,
        )
        if not self.encode_with_model:
            # numpy dict (np arrays)
            return dict(inputs)

        # Model-encoding path (PyTorch).
        import torch

        assert self.model is not None
        with torch.inference_mode():
            # Ensure inputs are on the same device as the model.
            # By default the model stays on CPU; callers may move it to CUDA if desired.
            device = next(self.model.parameters()).device
            model_inputs = {k: v.to(device) for k, v in inputs.items()}
            out = self.model(**model_inputs).last_hidden_state

        # Return numpy for compatibility with the existing create_tasks() path.
        return out.detach().cpu().numpy()


class MuseEmbedding(TextProcessor):
    def __init__(self):
        import tensorflow_hub as hub  # lazy import
        import tensorflow_text  # noqa: F401
        import tensorflow as tf  # lazy import

        self._tf = tf

        self.muse_model = hub.load(MULTI_MODULE)

    def encode(self, strings: Sequence[str]):
        # TF is imported lazily to allow torch-only installs.
        with self._tf.device("/cpu:0"):
            return self.muse_model(strings).numpy()


class CLIPTextProcessor(TextProcessor):
    def __init__(
        self,
        tokenizer_kwargs: Optional[dict] = {
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
    ):
        from transformers import CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.kwargs = tokenizer_kwargs

    def encode(self, strings: Sequence[str]):
        inputs = self.processor(
            text=strings,
            **self.kwargs,
        )
        inputs["position_ids"] = np.expand_dims(
            np.arange(inputs["input_ids"].shape[1]), axis=0
        ).repeat(inputs["input_ids"].shape[0], axis=0)
        return inputs
