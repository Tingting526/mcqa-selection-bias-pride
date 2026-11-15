"""Check pinned model access and fetch tokenizer files only; no weights/inference."""
import os
os.environ["USE_TORCH"] = "0"
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download, get_token

plan_path, output = map(Path, sys.argv[1:3])
plan = json.loads(plan_path.read_text())
rows = []
for model, config in plan["models"].items():
    try:
        path = snapshot_download(config["tokenizer_repo"], revision=config["tokenizer_revision"],
            allow_patterns=["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                            "added_tokens.json", "vocab.json", "merges.txt", "config.json"],
            cache_dir=str(output.parent / "tokenizer_cache"))
        row = dict(model=model, status="tokenizer_accessible", snapshot=path)
    except Exception as error:
        row = dict(model=model, status="access_failed", error_type=type(error).__name__,
                   http_status=getattr(getattr(error, "response", None), "status_code", None))
    rows.append(row)
    print(json.dumps(row), flush=True)
output.write_text(json.dumps(dict(authenticated_hf_token_available=bool(get_token()), models=rows), indent=2) + "\n")
