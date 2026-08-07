"""Load and format the smoke-test SFT dataset (smol-smoltalk subset)."""
from datasets import DatasetDict, load_dataset

TAGS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def format_messages(messages: list[dict]) -> str:
    parts = [f"{TAGS[m['role']]}\n{m['content']}" for m in messages]
    return "\n".join(parts) + "\n<|end|>"


def load_smoltalk(n_train: int = 4000, n_eval: int = 200, seed: int = 0) -> DatasetDict:
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    ds = ds.shuffle(seed=seed).select(range(n_train + n_eval))
    ds = ds.map(
        lambda ex: {"text": format_messages(ex["messages"])},
        remove_columns=ds.column_names,
    )
    return DatasetDict(
        train=ds.select(range(n_train)),
        eval=ds.select(range(n_train, n_train + n_eval)),
    )
