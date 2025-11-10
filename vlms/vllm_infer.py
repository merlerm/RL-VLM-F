import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

from PIL import Image
from vllm import SamplingParams

from vlms.vllm_engine import VllmModel, format_image

import numpy as np


DEFAULT_MODEL_ID = os.environ.get("VLLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")

_MODEL: Optional[VllmModel] = None
_MODEL_ID: Optional[str] = None


def _build_messages_from_query(query_list: Union[Sequence[object], object]) -> List[dict]:
    if isinstance(query_list, Sequence) and not isinstance(query_list, (str, bytes, os.PathLike)):
        parts = list(query_list)
    else:
        parts = [query_list]

    content: List[dict] = []
    for part in parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
            continue

        if isinstance(part, (os.PathLike, Path)):
            part = Path(part)
            content.append({"type": "image_url", "image_url": {"url": format_image(part)}})
            continue

        if isinstance(part, Image.Image):
            content.append({"type": "image_url", "image_url": {"url": format_image(part)}})
            continue

        if isinstance(part, np.ndarray):
            img = Image.fromarray(part.astype("uint8"))
            content.append({"type": "image_url", "image_url": {"url": format_image(img)}})
            continue

        raise TypeError(f"Unsupported query component: {type(part)!r}")

    return [{"role": "user", "content": content}]


def _make_sampling_params(
    model: VllmModel,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    best_of: Optional[int] = None,
    seed: Optional[int] = None,
    stop_token_ids: Optional[List[int]] = None,
) -> SamplingParams:
    base = model.sampling_params
    n_value = num_return_sequences if num_return_sequences is not None else getattr(base, "n", 1)
    best_of_value = best_of if best_of is not None else getattr(base, "best_of", n_value)
    return SamplingParams(
        seed=seed if seed is not None else getattr(base, "seed", None),
        max_tokens=max_tokens if max_tokens is not None else getattr(base, "max_tokens", 512),
        min_tokens=min_tokens if min_tokens is not None else getattr(base, "min_tokens", 1),
        temperature=temperature if temperature is not None else getattr(base, "temperature", 0.8),
        top_p=top_p if top_p is not None else getattr(base, "top_p", 0.95),
        top_k=top_k if top_k is not None else getattr(base, "top_k", 50),
        n=n_value,
        best_of=best_of_value,
        stop_token_ids=stop_token_ids,
    )


def _get_model(
    model: Optional[VllmModel] = None,
    *,
    model_id: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
) -> VllmModel:
    global _MODEL, _MODEL_ID
    if model is not None:
        return model

    target_id = model_id or DEFAULT_MODEL_ID
    if _MODEL is not None and _MODEL_ID == target_id:
        return _MODEL

    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("model_id", target_id)
    _MODEL = VllmModel(**kwargs)
    _MODEL_ID = _MODEL.model_id
    return _MODEL


def _run_chat(
    messages: List[dict],
    *,
    model: VllmModel,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    best_of: Optional[int] = None,
    seed: Optional[int] = None,
    stop_token_ids: Optional[List[int]] = None,
) -> Optional[str]:
    params = _make_sampling_params(
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        best_of=best_of,
        seed=seed,
    stop_token_ids=stop_token_ids,
    )

    outputs = model.model.chat(messages, sampling_params=params, use_tqdm=model.verbose)

    first = outputs[0]
    if not first.outputs:
        return None
    return first.outputs[0].text


def _last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def vllm_query_1(
    query_list: Union[Sequence[object], object],
    *,
    temperature: float = 0.0,
    model: Optional[VllmModel] = None,
    model_id: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
    return_full_text: bool = False,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    best_of: Optional[int] = None,
    seed: Optional[int] = None,
) -> Union[str, int]:
    model_instance = _get_model(model=model, model_id=model_id, model_kwargs=model_kwargs)
    messages = _build_messages_from_query(query_list)

    start = time.time()
    text = _run_chat(
        messages,
        model=model_instance,
        temperature=temperature,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        best_of=best_of,
        seed=seed,
    )
    elapsed = time.time() - start
    print(f"time elapsed: {elapsed:.2f}s")

    if text is None:
        return -1

    if return_full_text:
        return text.strip()
    return _last_non_empty_line(text)


def vllm_query_2(
    query_list: Union[Sequence[object], object],
    summary_prompt: str,
    *,
    temperature: float = 0.0,
    summary_temperature: Optional[float] = None,
    model: Optional[VllmModel] = None,
    model_id: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
    return_intermediate: bool = False,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    best_of: Optional[int] = None,
    seed: Optional[int] = None,
    summary_max_tokens: Optional[int] = None,
    summary_min_tokens: Optional[int] = None,
    summary_top_p: Optional[float] = None,
    summary_top_k: Optional[int] = None,
    summary_num_return_sequences: Optional[int] = None,
    summary_best_of: Optional[int] = None,
    summary_seed: Optional[int] = None,
) -> Union[str, int, tuple]:
    model_instance = _get_model(model=model, model_id=model_id, model_kwargs=model_kwargs)
    messages = _build_messages_from_query(query_list)

    start = time.time()
    analysis_text = _run_chat(
        messages,
        model=model_instance,
        temperature=temperature,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        best_of=best_of,
        seed=seed,
    )

    if analysis_text is None:
        return -1

    summary_prompt_text = summary_prompt.format(analysis_text)
    summary_messages = _build_messages_from_query([summary_prompt_text])

    summary_text = _run_chat(
        summary_messages,
        model=model_instance,
        temperature=summary_temperature if summary_temperature is not None else temperature,
        max_tokens=summary_max_tokens if summary_max_tokens is not None else max_tokens,
        min_tokens=summary_min_tokens if summary_min_tokens is not None else min_tokens,
        top_p=summary_top_p if summary_top_p is not None else top_p,
        top_k=summary_top_k if summary_top_k is not None else top_k,
        num_return_sequences=summary_num_return_sequences if summary_num_return_sequences is not None else num_return_sequences,
        best_of=summary_best_of if summary_best_of is not None else best_of,
        seed=summary_seed if summary_seed is not None else seed,
    )

    elapsed = time.time() - start
    print(f"time elapsed: {elapsed:.2f}s")

    if summary_text is None:
        return -1

    summary_line = _first_non_empty_line(summary_text)
    if return_intermediate:
        return summary_line, analysis_text.strip()
    return summary_line


if __name__ == "__main__":
    from prompt import (
        gemini_free_query_env_prompts,
        gemini_free_query_prompt1,
        gemini_free_query_prompt2,
        gemini_summary_env_prompts,
    )
    from PIL import Image as _Image

    env_name = "metaworld_sweep-into-v2"
    image_1_path = "data/images/metaworld_sweep-into-v2/image_6_1.png"
    image_2_path = "data/images/metaworld_sweep-into-v2/image_6_2.png"

    image_1 = _Image.open(image_1_path)
    image_2 = _Image.open(image_2_path)

    query_sequence = [
        gemini_free_query_prompt1,
        image_1,
        gemini_free_query_prompt2,
        image_2,
        gemini_free_query_env_prompts[env_name],
    ]

    result = vllm_query_2(
        query_sequence,
        gemini_summary_env_prompts[env_name],
        temperature=0.0,
        return_intermediate=True,
    )
    print(result)
