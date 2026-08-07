"""Sustained GPU burn: large bf16 matmuls while logging thermals.

Verifies the bring-up thermal setup under load. This box's vBIOS rejects
`nvidia-smi -pl`, so there is no power cap — this measurement decides whether a
clock cap (`nvidia-smi -lgc`) is needed instead.

Logs CPU package temp too: on the GS66 the CPU and GPU share the cooling
solution, so a GPU-only reading understates the thermal picture.

Usage: uv run scripts/burn.py [minutes]
"""
import subprocess
import sys
import time

import torch

GPU_Q = "temperature.gpu,power.draw,clocks.sm,utilization.gpu"


def gpu_stats() -> str:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def cpu_temp() -> str:
    out = subprocess.run(["sensors"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("Package id 0:"):
            return line.split()[3]
    return "n/a"


def main(minutes: float) -> None:
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    gpu_stats()  # discard: this GPU's first power.draw sample reads ~751 W

    peak_gpu = peak_cpu = 0.0
    end = time.time() + minutes * 60
    i = 0
    while time.time() < end:
        a = (a @ b).tanh()  # tanh keeps values bounded so this can run forever
        i += 1
        if i % 200 == 0:
            torch.cuda.synchronize()
            g, c = gpu_stats(), cpu_temp()
            peak_gpu = max(peak_gpu, float(g.split(",")[0]))
            try:
                peak_cpu = max(peak_cpu, float(c.rstrip("C+°")))
            except ValueError:
                pass
            mem = torch.cuda.memory_reserved() / 2**30
            print(f"[{time.strftime('%H:%M:%S')}] iter={i:6d} gpu={g} "
                  f"cpu={c} vram_reserved={mem:.2f}GiB", flush=True)

    torch.cuda.synchronize()
    print(f"\ndone after {i} iters. peak GPU {peak_gpu:.0f}C, peak CPU {peak_cpu:.0f}C")
    print("final:", gpu_stats())
    verdict = "OK for unattended training" if peak_gpu < 87 else "TOO HOT - apply clock cap"
    print("verdict:", verdict)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
