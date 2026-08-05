# Lab-box bring-up checklist — MSI GS66 12UGS (`gs66-lab`)

Manual runbook for Arley. Est. 1–2 h hands-on (plus download time). When every
box in the Verification section ticks, the machine is ready for the phase-0
plan (`docs/superpowers/plans/2026-08-05-phase-0-smoke-test.md`).

Two decisions are embedded below: **D1** (wipe vs. dual-boot) at step 3,
**D2** (repo transfer) at step 10.

## 1. While Windows still boots (optional, one-time chance)

- [ ] Update BIOS/EC firmware via MSI Center — much easier from Windows, and
      firmware bugs on this platform affect suspend/thermals. Skippable if it
      already runs fine.

## 2. BIOS (press `Del` during boot)

- [ ] **Disable Secure Boot** — avoids the NVIDIA kernel-module signing dance
      entirely.
- [ ] If present: set *AC Power Loss* / *Restore on AC* to power on — a
      headless box should come back after an outage.

## 3. Install Fedora 44 Workstation

**D1 — recommended: wipe the whole disk.** This is a dedicated lab box; a
Windows partition just eats the 1 TB SSD. (Alternative: shrink Windows and
dual-boot — only if you want to keep the machine's Windows licence usable for
something else. Costs ~150+ GB.)

- [ ] On the Zenbook, write the Fedora 44 Workstation ISO to a USB stick:
      Fedora Media Writer, or
      `sudo dd if=Fedora-Workstation-Live-x86_64-44-*.iso of=/dev/sdX bs=8M status=progress oflag=direct`
- [ ] Boot the stick (`F11` for the MSI boot menu), install with automatic
      partitioning over the entire disk (Btrfs defaults are fine).
- [ ] Hostname: `gs66-lab`. User: `arley` (administrator).
- [ ] First boot:
      `sudo dnf upgrade --refresh -y && sudo systemctl reboot`

## 4. NVIDIA driver (RPM Fusion)

```bash
sudo dnf install -y \
  https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
  https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
```

- [ ] **Wait for the module build** before rebooting: run
      `modinfo -F version nvidia` every minute until it prints a version
      (~5 min), then reboot.
- [ ] Verify: `nvidia-smi` shows "RTX 3070 Ti Laptop GPU".

## 5. SSH + headless

```bash
sudo systemctl enable --now sshd
sudo systemctl set-default multi-user.target   # no desktop = no VRAM stolen
```

- [ ] From the Zenbook: `ssh-copy-id arley@gs66-lab.local` (if `.local`
      doesn't resolve, get the IP with `ip -4 addr` on the lab box).
- [ ] Once key login works, disable password auth:
      `printf 'PasswordAuthentication no\n' | sudo tee /etc/ssh/sshd_config.d/90-keys-only.conf && sudo systemctl reload sshd`
- [ ] Reboot; confirm you can SSH in with the lid closed-ish (next step makes
      that safe).

## 6. Lid & suspend behaviour (critical — headless laptop)

```bash
sudo mkdir -p /etc/systemd/logind.conf.d
printf '[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nHandleLidSwitchDocked=ignore\n' \
  | sudo tee /etc/systemd/logind.conf.d/lab.conf
sudo systemctl restart systemd-logind
```

- [ ] Close the lid, wait 2 min, SSH in from the Zenbook — must still respond.

## 7. GPU power cap (~15% under the ~105 W TGP)

```bash
nvidia-smi -q -d POWER        # note "Max Power Limit" / whether limit is adjustable
sudo tee /etc/systemd/system/nvidia-powercap.service <<'EOF'
[Unit]
Description=Cap NVIDIA GPU power for sustained headless training
[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 90
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now nvidia-powercap.service
nvidia-smi --query-gpu=power.limit --format=csv   # expect 90.00 W
```

- [ ] If the vBIOS rejects `-pl` ("not supported"), skip the cap — the
      fallback is a clock cap (`nvidia-smi -lgc 300,1800` in the same unit).
      Either way, the phase-0 burn test (plan Task 2) measures what actually
      happens thermally.

## 8. Battery care (always-plugged laptop)

- [ ] If `/sys/class/power_supply/BAT1/charge_control_end_threshold` exists
      (msi-ec driver), cap charge:
      `echo 80 | sudo tee /sys/class/power_supply/BAT1/charge_control_end_threshold`
      and persist it:
      `printf 'w /sys/class/power_supply/BAT1/charge_control_end_threshold - - - - 80\n' | sudo tee /etc/tmpfiles.d/battery-cap.conf`
      If the file doesn't exist, skip — not all GS66 firmwares expose it.

## 9. Tooling

```bash
sudo dnf install -y git tmux btop nvtop
sudo dnf install -y uv || curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://claude.ai/install.sh | bash    # Claude Code, then: claude login
```

## 10. Repo transfer — D2

**Option A (done 2026-08-05): private repo at
`github.com/ArleyRistar/forgetting-lab`.** On the lab box:

```bash
sudo dnf install -y gh
gh auth login          # browser flow, ArleyRistar account, HTTPS protocol
gh repo clone ArleyRistar/forgetting-lab ~/forgetting-lab
```

**Option B (LAN only, no remote):** on the Zenbook:
`cd ~/personal/forgetting-lab && git bundle create /tmp/flab.bundle main && scp /tmp/flab.bundle arley@gs66-lab.local:~`
then here: `git clone -b main ~/flab.bundle ~/forgetting-lab`.

## 11. Physical

- [ ] Plugged in, chassis elevated (rear feet / stand) for airflow, somewhere
      the fan noise won't matter overnight.

## Verification — all must pass

- [ ] `nvidia-smi` shows the GPU and (if step 7 worked) `power.limit` = 90 W
- [ ] SSH from Zenbook works with the lid closed, no password prompt
- [ ] `systemctl get-default` → `multi-user.target`
- [ ] `uv --version` and `claude --version` both print
- [ ] `~/forgetting-lab` cloned; `git log --oneline` shows the spec commits
- [ ] Machine survives 10 min lid-closed idle without suspending

Then, on the lab box: `cd ~/forgetting-lab && claude`, and point the session at
`docs/superpowers/plans/2026-08-05-phase-0-smoke-test.md`.
