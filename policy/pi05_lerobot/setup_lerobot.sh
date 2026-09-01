#!/bin/bash
# ----------------------------------------------------------------------------
# Clone LeRobot into policy/pi05_lerobot/lerobot and install it editable in the
# current Python environment, with the extras PI0.5 needs.
#
#   bash policy/pi05_lerobot/setup_lerobot.sh [install_dir]
#
# LEROBOT_REF defaults to a pinned upstream commit that contains MEM
# (huggingface/lerobot#4076). Pass LEROBOT_REF=main to track the moving tip.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$SCRIPT_DIR/lerobot}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
# huggingface/lerobot main @ 2026-08-31. Contains #4076 (visual + proprioceptive
# MEM, src/lerobot/policies/pi05/memory.py) and #4056 (training-time RTC).
LEROBOT_REF="${LEROBOT_REF:-d36d404b65315139b7601a707f260a3db736462f}"
# pi -> transformers + scipy for PI0.5; training -> dataset + wandb + accelerate.
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-pi,training}"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "LeRobot already exists: $INSTALL_DIR"
else
    git clone "$LEROBOT_REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
git fetch origin "$LEROBOT_REF" || git fetch origin
git checkout "$LEROBOT_REF"

if [[ ! -f src/lerobot/policies/pi05/memory.py ]]; then
    echo "Error: this LeRobot revision has no src/lerobot/policies/pi05/memory.py," >&2
    echo "so it predates MEM. Pick a newer LEROBOT_REF." >&2
    exit 1
fi

python -m pip install -U pip
python -m pip install -e ".[${LEROBOT_EXTRAS}]"

echo "LeRobot is ready at: $INSTALL_DIR ($(git rev-parse --short HEAD))"
echo "For eval and finetune, set:"
echo "  export PI05_LEROBOT_ROOT=$INSTALL_DIR"
