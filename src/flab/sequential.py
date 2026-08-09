"""Sequential fine-tuning harness (spec §6 1a, phase-1a task 4).

base -> task A -> task B -> ..., probing held-out NLL at every stage boundary
and at a baseline before stage 0. The baseline matters more than it looks: spec
§9 requires forgetting to be measured against *each model's own* post-conversion
baseline rather than as a cross-model absolute, because the ternary twin starts
weaker than its float twin. Taking a baseline unconditionally makes the correct
comparison the easy one.

**Crash ordering.** A stage is marked DONE only once its boundary probe is on
disk. Marking DONE at the end of training instead would mean a crash in the
gap between training and probing loses that boundary permanently — the stage
would be skipped on resume and no error would ever be raised. Training is still
not repeated in that window: the saved checkpoint is recorded first, so a
resume finds the weights, skips the training, and goes straight to probing.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch

from flab import probes, trace
from flab.runconfig import RunConfig
from flab.runstate import RunState, DONE, RUNNING


def stage_dir(run_dir: Path, index: int, task: str) -> Path:
    return run_dir / f"stage-{index}-{task}"


# -- model plumbing ------------------------------------------------------


def verify_model_digest(cfg: RunConfig) -> str | None:
    """Check the downloaded weights against the digest pinned in the config.

    Llama 3.2 1B is gated upstream and we train on a mirror whose weights are
    byte-identical to Meta's release. "Byte-identical today" is not a property
    that maintains itself, and a mirror that quietly diverges would produce a
    calibration against unknown weights with nothing in the output to show it.
    Raises rather than warns: a provenance failure is not a degraded run.
    """
    if not cfg.model_sha256:
        return None
    import hashlib

    from huggingface_hub import try_to_load_from_cache

    # Repos do not agree on the weight filename: Llama and Gemma ship
    # `model.safetensors`, Qwen3.5-0.8B ships
    # `model.safetensors-00001-of-00001.safetensors`. Look for the standard
    # name, then fall back to the single safetensors file in the snapshot —
    # and refuse if that is ambiguous rather than hashing an arbitrary shard.
    path = try_to_load_from_cache(cfg.model, "model.safetensors")
    if not isinstance(path, str):
        anchor = try_to_load_from_cache(cfg.model, "config.json")
        if not isinstance(anchor, str):
            raise RuntimeError(f"cannot verify {cfg.model}: nothing in cache")
        found = sorted(Path(anchor).parent.glob("*.safetensors*"))
        if len(found) != 1:
            raise RuntimeError(
                f"cannot verify {cfg.model}: expected one safetensors file, "
                f"found {len(found)} — pin a digest per shard or drop model_sha256")
        path = str(found[0])
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != cfg.model_sha256:
        raise RuntimeError(
            f"{cfg.model} weights do not match the pinned digest.\n"
            f"  expected {cfg.model_sha256}\n  got      {got}\n"
            "Refusing to train on unverified weights."
        )
    return got


def _load_base(cfg: RunConfig):
    from transformers import AutoModelForCausalLM

    verify_model_digest(cfg)
    # transformers 5.x renamed torch_dtype -> dtype (LAB-NOTES quirk 2)
    kwargs = {"dtype": "bfloat16"} if torch.cuda.is_available() else {}
    model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
    return model.cuda() if torch.cuda.is_available() else model


def checkpoint_ok(path: str | Path | None, mode: str) -> bool:
    """Is this a checkpoint we can actually load?

    Directory-exists is not enough. A crash *during* `save_model` leaves the
    directory there but incomplete, and resume would then die inside peft with
    a message about a missing adapter_config rather than simply retraining the
    stage. Treat an unloadable checkpoint as no checkpoint.
    """
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    if mode == "lora":
        return (p / "adapter_config.json").is_file()
    return (p / "model.safetensors").is_file() or any(p.glob("model-*.safetensors"))


def _latest_checkpoint(cfg: RunConfig, state: RunState) -> str | None:
    """The most recent stage whose weights actually reached disk, intact.

    Not simply the last DONE stage: a stage that finished training but crashed
    before its probe has a checkpoint and is not DONE, and its weights are
    exactly what the resumed run must continue from.
    """
    latest = None
    for s in state.stages:
        if checkpoint_ok(s.checkpoint, cfg.mode):
            latest = s.checkpoint
    return latest


def _prepare_model(cfg: RunConfig, model, state: RunState):
    """Attach LoRA (or not) and restore whatever the last stage left behind.

    LoRA and full mode deliberately converge here rather than forking, so the
    probe and the stage loop below see the same object either way. Phase 1c
    runs a ternary full fine-tune through this same path — the fork would
    otherwise be discovered at the worst possible moment.
    """
    ckpt = _latest_checkpoint(cfg, state)

    if cfg.mode == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model

        if ckpt:
            # One adapter is carried across the whole run: sequential
            # fine-tuning trains the *same* adapter stage after stage. Starting
            # a fresh adapter per stage would be a different experiment.
            return PeftModel.from_pretrained(model, ckpt, is_trainable=True)
        tm = cfg.lora.target_modules
        return get_peft_model(model, LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            # peft takes the literal string "all-linear" or a list of names.
            target_modules=tm if isinstance(tm, str) else list(tm),
        ))

    if ckpt:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(Path(ckpt) / "model.safetensors"), strict=False)
    return model


# -- stages --------------------------------------------------------------


def _train_stage(cfg: RunConfig, model, tokenizer, index: int, out: Path) -> int:
    from trl import SFTConfig, SFTTrainer

    stage = cfg.stages[index]
    cuda = torch.cuda.is_available()
    batch, accum = (cfg.train.batch_size, cfg.train.grad_accum) if cuda else (1, 1)

    # With max_steps, prepare only what the stage consumes: the example order is
    # a deterministic hash of (seed, index), so the first N of a 5000-row pool
    # and a pool of exactly N are the *same* N examples — a cost saving, not a
    # change of experiment. With epochs, "the whole split" is the definition, so
    # take all of it.
    n_train = None if stage.epochs else stage.max_steps * batch * accum
    data = trace.load_task(
        stage.task, n_train=n_train, n_eval=8, seed=cfg.seed,
        tokenizer=tokenizer, max_length=cfg.train.max_length,
        variant=cfg.trace_variant,
    )
    resume = out.exists() and any(out.glob("checkpoint-*"))
    steps = min(20, max(1, (stage.max_steps or 100) // 10))

    args = SFTConfig(
        output_dir=str(out),
        max_steps=stage.max_steps if stage.max_steps else -1,
        num_train_epochs=stage.epochs if stage.epochs else 3.0,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=stage.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_steps=cfg.train.warmup_steps if cfg.train.warmup_steps is not None else steps,
        bf16=cuda,
        gradient_checkpointing=cuda,
        max_length=cfg.train.max_length,
        logging_steps=10,
        save_steps=50,          # mid-stage crash insurance...
        save_total_limit=2,
        eval_strategy="no",     # ...but never speculatively evaluated (spec §6 1a).
        optim=cfg.optim if cuda else "adamw_torch",
        report_to=[],
        seed=cfg.seed,
    )
    # Pass the tokenizer explicitly: without it TRL re-derives one from
    # model.config._name_or_path, which is empty for any model built in
    # memory rather than loaded from the hub.
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=data["train"], processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out))
    return int(trainer.state.global_step)


def _probe_to_disk(cfg: RunConfig, model, tokenizer, run_dir: Path, name: str) -> str:
    from dataclasses import asdict

    result = probes.probe_all(
        model, tokenizer, cfg.probe_tasks,
        n_eval=cfg.probe.n_eval, max_length=cfg.probe.max_length,
        batch_size=cfg.probe.batch_size, seed=cfg.seed, variant=cfg.trace_variant,
    )
    if cfg.probe.reference_n:
        # Drift on a set the model never trains on. Cheap, and the only metric
        # here that phase 1d's flip-fraction has a direct analogue to.
        stab = probes.probe_stability(
            model, tokenizer, n_ref=cfg.probe.reference_n,
            max_length=cfg.probe.max_length, batch_size=cfg.probe.batch_size,
            seed=cfg.seed, variant=cfg.trace_variant,
        )
        result["stability"] = asdict(stab)
        if stab.warning:
            result["warnings"] = {**(result.get("warnings") or {}), "stability": stab.warning}
    (run_dir / name).write_text(json.dumps(result, indent=2))
    return name


# -- entrypoint ----------------------------------------------------------


def run(cfg: RunConfig, run_dir: str | Path, model=None, tokenizer=None) -> RunState:
    from transformers import AutoTokenizer

    run_dir = Path(run_dir)
    state = RunState.create_or_load(run_dir, cfg)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model)
    model = _prepare_model(cfg, model if model is not None else _load_base(cfg), state)

    if state.baseline_probe is None:
        state.set_baseline(_probe_to_disk(cfg, model, tokenizer, run_dir, "probe-baseline.json"))

    while (i := state.next_index()) is not None:
        stage, st = cfg.stages[i], state.stages[i]
        out = stage_dir(run_dir, i, stage.task)

        if not checkpoint_ok(st.checkpoint, cfg.mode):
            state.mark(i, RUNNING)
            steps = _train_stage(cfg, model, tokenizer, i, out)
            # Record the checkpoint *before* probing, so a crash in the gap
            # resumes into the probe rather than retraining the stage.
            state.mark(i, RUNNING, steps_done=steps, checkpoint=str(out))

        probe_name = _probe_to_disk(cfg, model, tokenizer, run_dir, f"probe-after-{i}.json")
        state.mark(i, DONE, probe=probe_name)

    # Convention: the marker means *finished*, not *worked*.
    (run_dir / "COMPLETE").write_text(f"stages={len(cfg.stages)}\n")
    return state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dev-3stage.yaml")
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()
    cfg = RunConfig.load(args.config)
    run(cfg, args.run_dir or Path("outputs/runs") / cfg.run_name)


if __name__ == "__main__":
    main()
