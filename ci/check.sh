#!/usr/bin/env bash
# Run before opening or updating a PR: the same checks CI runs.
#   ci/check.sh            # tests, viewer build, version bump vs origin/main
#   ci/check.sh --no-release
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests
tmp=$(mktemp -d)
python -m implicit.build examples/*.py --out "$tmp/site" > /dev/null
if node -e "require('playwright')" 2> /dev/null; then
  node ci/gpu_check.mjs "$tmp/site" swiftshader SwiftShader | tail -1   # GPU check (SwiftShader)
else
  echo "skipping GPU check: install Playwright (npm install playwright) to run it locally"
fi
rm -rf "$tmp"
if [[ "${1:-}" != "--no-release" ]]; then
  git fetch -q origin main
  python ci/release.py check origin/main
fi
echo "all checks passed"
