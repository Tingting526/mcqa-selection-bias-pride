#!/usr/bin/env bash
# Run on the Mac. Downloads existing results only; submits no jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
remote='<lrz-user>@login.ai.lrz.de:/dss/dsshome1/<group>/<lrz-user>/analysis/'
campaign='outputs/campaigns/medqa-three-models-20260930'
mkdir -p "$campaign"
rsync -av "${remote}${campaign}/" "$campaign/"
file_list=$(mktemp)
trap 'rm -f "$file_list"' EXIT
python3 -c '
import json, sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text())
for entry in state["entries"]:
    name = Path(entry["pipeline"]).name
    for kind in ("baseline", "pride", "permutations"):
        print(f"outputs/raw/{kind}/{name}/")
' "$campaign/campaign.json" > "$file_list"
rsync -avr --files-from="$file_list" "$remote" ./
echo 'MedQA-Ergebnisse und Permutationen heruntergeladen. Notebooks 09 und 10 sind bereit.'
