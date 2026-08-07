"""LoRA SFT smoke test: SmolLM2-360M on a smol-smoltalk subset (spec §5)."""
import argparse

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from flab.data import load_smoltalk

MODEL = "HuggingFaceTB/SmolLM2-360M"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/smoke")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    data = load_smoltalk()
    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        report_to="tensorboard",
        seed=0,
        # transformers 5.x renamed torch_dtype -> dtype
        model_init_kwargs={"dtype": "bfloat16"},
    )
    trainer = SFTTrainer(
        model=MODEL,
        args=cfg,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()
