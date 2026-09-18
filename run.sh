#!/usr/bin/env bash
# runs the given command on a gpu: this node if it has one, else one h200 through slurm
set -euo pipefail

test $# -gt 0 || { echo "usage: $0 <command...>" >&2 ; exit 1 ; }
cd "$(dirname "$(readlink -f "$0")")"

slurm_gpu() {
	mkdir -p logs
	# CACHE survives the job and is shared across nodes, $HOME is too small for 52 GiB of weights
	export ROOT=/n/netscratch/amin_lab/Lab/$USER
	export CACHE=$ROOT/cache HF_HOME=$ROOT/hf
	export UV_CACHE_DIR=$CACHE/uv UV_PYTHON_INSTALL_DIR=$CACHE/python TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
	mkdir -p "$CACHE" "$HF_HOME"
	trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

	JOB=$(sbatch --parsable -J qwen27b-jev -p seas_gpu --gres=gpu:nvidia_h200:1 -c 16 --mem=128G -t 4:00:00 -o logs/%j.out --wrap "$*")
	echo "submitted $JOB" >&2
	tail -F "logs/$JOB.out" >&2 &
	while squeue -h -j "$JOB" -o %i | grep -q .
	do
		sleep 5
	done
	sleep 2 # let tail -F flush the last lines before it dies
	kill %tail 2>/dev/null || true
}

if nvidia-smi -L 2>/dev/null | grep -q GPU
then
	exec "$@"
elif command -v sbatch >/dev/null
then
	slurm_gpu "$@"
else
	echo "no gpu and no slurm" >&2
	exit 1
fi
