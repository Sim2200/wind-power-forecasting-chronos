#!/bin/bash
# Download SDWPF (Baidu KDD Cup 2022, Zhou et al. 2024) into data/sdwpf/raw/ and verify checksums.
#
# Source: Hugging Face mirror `aigrids/WindFarm_raw`, file sdwpf_kddcup.tar.gz (1.0 GB). Besides the two CSVs
# below it holds the competition's hidden test set (final_phase_test.zip), which we do not use, so only the
# CSVs are extracted. The original source is Baidu AI Studio; the CSVs here are byte-identical to the ones used
# for results/ds01_counts.json (sha256 below).
#
# Usage: scripts/download_sdwpf.sh            (downloads the tarball, then deletes it)
#        SDWPF_TAR=/path/to/sdwpf_kddcup.tar.gz scripts/download_sdwpf.sh   (use a tarball you already have)
# Then:  python -m wpf.data.sdwpf
set -eo pipefail
DEST="$(dirname "$0")/../data/sdwpf/raw"
URL="https://huggingface.co/datasets/aigrids/WindFarm_raw/resolve/main/sdwpf_kddcup.tar.gz"
TAR_SHA256="3ad8dd27d065fcd939fedd6944c0f032b446ee60881429c79d25b6c0b4e77b79"
DATA_CSV="sdwpf_245days_v1.csv"  # 4,727,520 rows: 134 turbines x 245 days x 144 ten-minute readings
DATA_SHA256="6cf7613a82c61bb5cf9fcbbf22e3f6c48165fa32021eb6f1905cb2b7af4d7667"
LOC_CSV="sdwpf_baidukddcup2022_turb_location.csv"  # TurbID, x, y
LOC_SHA256="15c19ceca43171dfda06e8de487d4d55eb116112904d77958aed7260c2f0d9fe"

# Linux has sha256sum; macOS has shasum.
if command -v sha256sum >/dev/null; then CHECK=(sha256sum -c -); else CHECK=(shasum -a 256 -c -); fi

mkdir -p "$DEST"
if [ -n "$SDWPF_TAR" ]; then
  TAR="$SDWPF_TAR"
else
  TAR="$DEST/sdwpf_kddcup.tar.gz"
  curl -sSL --fail -o "$TAR" "$URL"
fi
echo "$TAR_SHA256  $TAR" | "${CHECK[@]}"

# The tarball is fed on stdin: GNU tar reads a path like C:/... (Git Bash on Windows) as a remote host.
tar -xz -C "$DEST" --strip-components=1 "sdwpf_kddcup/$DATA_CSV" "sdwpf_kddcup/$LOC_CSV" < "$TAR"
echo "$DATA_SHA256  $DEST/$DATA_CSV" | "${CHECK[@]}"
echo "$LOC_SHA256  $DEST/$LOC_CSV" | "${CHECK[@]}"

if [ -z "$SDWPF_TAR" ]; then rm "$TAR"; fi
