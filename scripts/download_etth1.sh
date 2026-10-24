#!/bin/bash
# Download ETTh1.csv (ETDataset, Zhou et al. 2021) into data/etth1/ and verify its checksum.
set -eo pipefail
DEST="$(dirname "$0")/../data/etth1"
URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
SHA256="f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"

mkdir -p "$DEST"
curl -sSL -o "$DEST/ETTh1.csv" "$URL"
if command -v sha256sum >/dev/null; then
  echo "$SHA256  $DEST/ETTh1.csv" | sha256sum -c -
else
  echo "$SHA256  $DEST/ETTh1.csv" | shasum -a 256 -c -   # macOS
fi
