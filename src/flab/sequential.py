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

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def stage_dir(run_dir: Path, index: int, task: str) -> Path:
    return run_dir / f"stage-{index}-{task}"


# -- model plumbing ------------------------------------------------------


def _load_base(cfg: RunConfig):
    from transformers import AutoModelForCausalLM

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
        return get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
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
    batch, accum = (4, 4) if cuda else (1, 1)
    # Prepare only what the stage will actually consume. The example order is a
    # deterministic hash of (seed, index), so the first N of a 5000-row pool and
    # a pool of exactly N are the *same* N examples — this is a cost saving, not
    # a change of experiment. If N exceeds the split, load_task clamps and the
    # trainer cycles epochs as before.
    data = trace.load_task(
        stage.task, n_train=stage.max_steps * batch * accum, n_eval=8, seed=cfg.seed,
        tokenizer=tokenizer, max_length=cfg.probe.max_length,
    )
    resume = out.exists() and any(out.glob("checkpoint-*"))

    args = SFTConfig(
        output_dir=str(out),
        max_steps=stage.max_steps,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=stage.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=min(20, max(1, stage.max_steps // 10)),
        bf16=cuda,
        gradient_checkpointing=cuda,
        max_length=cfg.probe.max_length,
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
    result = probes.probe_all(
        model, tokenizer, cfg.probe_tasks,
        n_eval=cfg.probe.n_eval, max_length=cfg.probe.max_length,
        batch_size=cfg.probe.batch_size, seed=cfg.seed,
    )
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
