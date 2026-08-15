#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: stage_rxrx1_full.sh download|extract}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
IMAGE_URL=https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip
METADATA_URL=https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip
IMAGE_BYTES=49039640485
METADATA_BYTES=1333687
IMAGE_MD5=f75ffa93b917e8a82e27e0c3a2b3425f
METADATA_MD5=777e959ca0bce5a2aa7f9f9d456aee7e

mkdir -p "$DATA_ROOT"

verify_file() {
  local path=$1 expected_bytes=$2 expected_md5=$3
  test "$(stat -c %s "$path")" = "$expected_bytes"
  test "$(md5sum "$path" | awk '{print $1}')" = "$expected_md5"
}

download_file() {
  local url=$1 final=$2 expected_bytes=$3 expected_md5=$4
  local partial=$final.part
  if [[ -f "$final" ]]; then
    verify_file "$final" "$expected_bytes" "$expected_md5"
    return
  fi
  curl --fail --location --continue-at - --retry 20 --retry-delay 5 \
    --retry-all-errors --output "$partial" "$url"
  verify_file "$partial" "$expected_bytes" "$expected_md5"
  mv "$partial" "$final"
}

case $MODE in
  download)
    download_file "$METADATA_URL" "$DATA_ROOT/rxrx1-metadata.zip" \
      "$METADATA_BYTES" "$METADATA_MD5"
    download_file "$IMAGE_URL" "$DATA_ROOT/rxrx1-images.zip" \
      "$IMAGE_BYTES" "$IMAGE_MD5"
    printf '{"state":"downloaded","image_bytes":%s,"image_md5":"%s","metadata_bytes":%s,"metadata_md5":"%s"}\n' \
      "$IMAGE_BYTES" "$IMAGE_MD5" "$METADATA_BYTES" "$METADATA_MD5" \
      >"$DATA_ROOT/DOWNLOAD_COMPLETE.json"
    ;;
  extract)
    verify_file "$DATA_ROOT/rxrx1-images.zip" "$IMAGE_BYTES" "$IMAGE_MD5"
    verify_file "$DATA_ROOT/rxrx1-metadata.zip" "$METADATA_BYTES" "$METADATA_MD5"
    if [[ ! -f "$DATA_ROOT/EXTRACTION_COMPLETE.json" ]]; then
      mkdir -p "$DATA_ROOT/extracted"
      unzip -q -o "$DATA_ROOT/rxrx1-images.zip" -d "$DATA_ROOT/extracted"
      unzip -q -o "$DATA_ROOT/rxrx1-metadata.zip" -d "$DATA_ROOT/extracted"
      test "$(find "$DATA_ROOT/extracted/rxrx1" -type f | wc -l)" = 753063
      test -f "$DATA_ROOT/extracted/rxrx1/metadata.csv"
      test -f "$DATA_ROOT/extracted/rxrx1/README.md"
      if [[ -e "$DATA_ROOT/rxrx1" ]]; then
        echo "$DATA_ROOT/rxrx1 already exists without a completion marker" >&2
        exit 1
      fi
      mv "$DATA_ROOT/extracted/rxrx1" "$DATA_ROOT/rxrx1"
      printf '{"state":"extracted","files":753063}\n' \
        >"$DATA_ROOT/EXTRACTION_COMPLETE.json"
    fi
    ;;
  *) echo "mode must be download|extract" >&2; exit 2 ;;
esac
