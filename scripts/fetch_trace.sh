#!/usr/bin/env bash
# Fetch and verify the TRACE benchmark archive (phase-1a task 1).
#
# TRACE has no HuggingFace mirror. The data ships as a single Google Drive zip
# published alongside the Jan-2024 repo (BeyonderXX/TRACE), which makes that
# link the provenance root of every phase-1/2 result and a single point of
# failure. This script downloads it, checksums it, and keeps an archival copy
# so a dead link later is an inconvenience rather than a lost phase.
#
# The file is over Drive's virus-scan threshold, so a plain GET returns an HTML
# interstitial instead of the zip - hence the confirm-token dance below.
set -euo pipefail
cd "$(dirname "$0")/.."

FILE_ID=1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV
DEST=data/trace
ZIP=data/TRACE-Benchmark.zip
ARCHIVE_COPY="$HOME/archive/TRACE-Benchmark.zip"

mkdir -p data "$(dirname "$ARCHIVE_COPY")"

if [ ! -s "$ZIP" ]; then
  echo "== downloading TRACE-Benchmark.zip from Google Drive =="
  COOKIES=$(mktemp)
  # First request: collect the confirm token and the session cookie.
  CONFIRM=$(curl -sc "$COOKIES" "https://drive.google.com/uc?export=download&id=$FILE_ID" \
    | grep -oE 'confirm=[0-9A-Za-z_-]+' | head -1 | cut -d= -f2 || true)
  if [ -n "$CONFIRM" ]; then
    URL="https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$FILE_ID"
  else
    # Newer Drive flow serves the file from a different host with a form POST.
    URL="https://drive.usercontent.google.com/download?id=$FILE_ID&export=download&confirm=t"
  fi
  curl -Lb "$COOKIES" -o "$ZIP" "$URL"
  rm -f "$COOKIES"
fi

# A dead or gated link yields an HTML error page, not a zip. Catch that here
# rather than letting unzip fail with something less legible.
if ! file "$ZIP" | grep -qi zip; then
  echo "!! Download is not a zip archive - the Drive link is probably dead or gated."
  echo "!! Got: $(file -b "$ZIP") ($(stat -c%s "$ZIP") bytes)"
  echo "!! STOP. Do not substitute third-party HuggingFace re-uploads of the"
  echo "!! constituent tasks; their schemas and provenance are not verified."
  exit 1
fi

echo "== sha256 =="
sha256sum "$ZIP" | tee data/TRACE-Benchmark.zip.sha256

if [ ! -d "$DEST" ]; then
  echo "== extracting to $DEST =="
  mkdir -p "$DEST"
  unzip -q "$ZIP" -d "$DEST"
fi

[ -f "$ARCHIVE_COPY" ] || { cp "$ZIP" "$ARCHIVE_COPY"; echo "== archived to $ARCHIVE_COPY =="; }

echo "== inventory =="
find "$DEST" -name '*.json' | sort | while read -r f; do
  n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f" 2>/dev/null || echo "?")
  printf "%-64s %s\n" "${f#"$DEST"/}" "$n"
done
