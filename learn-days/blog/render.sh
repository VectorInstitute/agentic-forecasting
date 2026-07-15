#!/usr/bin/env bash
# Render every blog post to a self-contained HTML file (images + CSS inlined),
# plus a top-level index.html table of contents. Requires only pandoc.
#
#   ./render.sh
#
# Output (git-ignored, regenerable): <post-dir>/index.html and ./index.html.
# Each output is fully portable — open or email it, images travel with it.
set -euo pipefail
cd "$(dirname "$0")"
css="assets/pandoc.css"

entries=()
for md in [0-9]*/post.md; do
  dir="$(dirname "$md")"
  title="$(sed -n 's/^# //p' "$md" | head -1)"
  # Run pandoc from inside the post dir so relative image paths
  # (images/… and ../../assets/…) resolve for --embed-resources.
  ( cd "$dir" && pandoc post.md \
      --standalone --embed-resources \
      --css "../$css" \
      --metadata title="$title" \
      --metadata lang=en \
      -o index.html )
  echo "rendered  $dir/index.html"
  entries+=("$dir|$title")
done

# Top-level table of contents linking the posts in order.
{
  printf '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
  printf '<meta name="viewport" content="width=device-width, initial-scale=1">'
  printf '<title>Agentic Forecasting — a blog series</title><style>'
  cat "$css"
  printf '</style></head><body>\n'
  printf '<h1>Agentic Forecasting</h1>\n'
  printf '<p>A technical blog series, spun out of the bootcamp learn-day lectures.</p>\n<ol>\n'
  for e in "${entries[@]}"; do
    printf '<li><a href="%s/index.html">%s</a></li>\n' "${e%%|*}" "${e#*|}"
  done
  printf '</ol></body></html>\n'
} > index.html
echo "rendered  index.html  (table of contents)"
