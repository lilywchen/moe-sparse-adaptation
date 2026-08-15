#!/usr/bin/env bash
set -euo pipefail

ORCD_DATA=${ORCD_DATA:-/home/l1ly/orcd/scratch/moe-batch-effect/rxrx1-huvec-primary-fold0}
NERSC_KEY=${NERSC_KEY:-/home/l1ly/orcd/scratch/moe-batch-effect/nersc-auth/nersc}
NERSC_HOST=${NERSC_HOST:-lilychen@dtn01.nersc.gov}
NERSC_DATA=${NERSC_DATA:-/global/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-huvec-primary-fold0}

test -f "$NERSC_KEY"
test -f "$NERSC_KEY-cert.pub"
SSH=(ssh -i "$NERSC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new "$NERSC_HOST")
"${SSH[@]}" "mkdir -p '$NERSC_DATA/rxrx1/images'"

transfer_group() {
  local origin=$1
  shift
  local source_root
  if [[ "$origin" == source ]]; then
    source_root=$ORCD_DATA/rxrx1/images
  else
    source_root=$ORCD_DATA/eval-target-stage/rxrx1/images
  fi
  tar -C "$source_root" -cf - "$@" \
    | "${SSH[@]}" tar -C "$NERSC_DATA/rxrx1/images" -xf -
}

pids=()
transfer_group source HUVEC-01 HUVEC-02 HUVEC-03 HUVEC-04 & pids+=("$!")
transfer_group source HUVEC-06 HUVEC-07 HUVEC-10 HUVEC-13 & pids+=("$!")
transfer_group source HUVEC-14 HUVEC-16 HUVEC-18 HUVEC-19 & pids+=("$!")
transfer_group source HUVEC-20 HUVEC-21 HUVEC-23 HUVEC-24 & pids+=("$!")
transfer_group target HUVEC-05 HUVEC-08 HUVEC-09 HUVEC-11 & pids+=("$!")
transfer_group target HUVEC-12 HUVEC-15 HUVEC-17 HUVEC-22 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more direct HUVEC transfers failed" >&2
  exit "$status"
fi
"${SSH[@]}" "find '$NERSC_DATA/rxrx1/images' -mindepth 1 -maxdepth 1 -type d -name 'HUVEC-*' | wc -l"
