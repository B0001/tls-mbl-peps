#!/bin/bash
# Provision a GCP VM, ship the repo, and launch the L=16 production run.
# Usage:  tools/cloud_run.sh [create|ship|launch|status|fetch|delete]
# The run is resumable (zarr stage markers), so a preempted/killed VM only
# loses at most one ladder rung per realization.
set -euo pipefail

GC=/opt/homebrew/share/google-cloud-sdk/bin/gcloud
PROJECT=tlsmbl-compute
ZONE=us-central1-c  # -a/-b were ZONE_RESOURCE_POOL_EXHAUSTED for c2d-standard-8 on 2026-07-22
VM=tlsmbl-bench
MACHINE=c2d-standard-8
CONFIG=configs/bench_L16_D4.yaml
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
create)
  $GC compute instances create $VM \
    --project=$PROJECT --zone=$ZONE --machine-type=$MACHINE \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=30GB
  ;;
ship)
  # Tar the repo *contents* (not the directory) and unpack into a fixed remote name,
  # so this works regardless of what the local checkout directory is called.
  tar -C "$REPO_DIR" --exclude='./.git' --exclude='./.venv' --exclude='./runs' \
      --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.egg-info' \
      -czf /tmp/tlsmbl.tgz .
  $GC compute scp /tmp/tlsmbl.tgz $VM:~ --project=$PROJECT --zone=$ZONE
  $GC compute ssh $VM --project=$PROJECT --zone=$ZONE --command='
    set -e
    sudo apt-get -qq update && sudo apt-get -qq install -y python3-pip curl git
    curl -LsSf https://astral.sh/uv/install.sh | sh
    mkdir -p tls-mbl-peps && tar xzf tlsmbl.tgz -C tls-mbl-peps
    cd tls-mbl-peps
    ~/.local/bin/uv venv --python 3.12 .venv
    ~/.local/bin/uv pip install -e ".[dev]" --python .venv/bin/python
    ~/.local/bin/uv pip install torch --index-url https://download.pytorch.org/whl/cpu --python .venv/bin/python
    .venv/bin/python -m pytest tests/unit/test_config.py -q'
  ;;
launch)
  $GC compute ssh $VM --project=$PROJECT --zone=$ZONE --command="
    cd tls-mbl-peps && nohup ~/.local/bin/uv run tlsmbl run $CONFIG > run.log 2>&1 &
    echo launched; sleep 3; tail -3 run.log"
  ;;
status)
  $GC compute ssh $VM --project=$PROJECT --zone=$ZONE \
    --command='cd tls-mbl-peps && tail -5 run.log 2>/dev/null; ls runs/*.zarr/realizations 2>/dev/null | head -40'
  ;;
fetch)
  $GC compute ssh $VM --project=$PROJECT --zone=$ZONE \
    --command='cd tls-mbl-peps && tar czf /tmp/results.tgz runs/ run.log'
  $GC compute scp $VM:/tmp/results.tgz "$REPO_DIR/runs/cloud-results.tgz" --project=$PROJECT --zone=$ZONE
  echo "results at runs/cloud-results.tgz"
  ;;
delete)
  $GC compute instances delete $VM --project=$PROJECT --zone=$ZONE --quiet
  ;;
*)
  echo "usage: $0 create|ship|launch|status|fetch|delete" >&2
  exit 1
  ;;
esac
