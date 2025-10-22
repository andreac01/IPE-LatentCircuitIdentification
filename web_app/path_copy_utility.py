import re
import shutil
from pathlib import Path

def copy_path_from_folder(folder_name: str):
    """
    Copy every file named:
      paths_{model}_{task}_..._cf{True|False}_*.pkl
    from `folder_name` into:
      ./data/{model}/{task}/{cf}/paths{metric}.pkl

    Rules:
    - Trailing timestamp "-YYYYMMDD_HHMMSS" is removed before parsing.
    - First token after "paths_" is the model, second is the task.
    - The first token starting with "cf" (e.g. "cfFalse") becomes the {cf} folder.
    - Metric is built from the trailing lowercase tokens before the cf segment.
      If no such tokens exist, the file is saved as "paths.pkl".
    """

    src_dir = Path(folder_name).expanduser().resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"❌ Source folder does not exist or is not a directory: {src_dir}")
        return

    pkl_files = list(src_dir.glob("paths_*.pkl"))
    if not pkl_files:
        print(f"ℹ️ No matching 'paths_*.pkl' files found in {src_dir}")
        return

    metric_token_pattern = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*$")
    copied = 0

    for src in pkl_files:
        stem_no_ts = re.sub(r"-\d{8}_\d{6}$", "", src.stem)
        core = stem_no_ts[len("paths_"):] if stem_no_ts.startswith("paths_") else stem_no_ts

        parts = core.split("_")
        if not parts:
            print(f"⚠️ Skipping unparsable filename: {src.name}")
            continue

        model = parts[0] if len(parts) >= 1 and parts[0] else "unknown_model"
        task = parts[1] if len(parts) >= 2 and parts[1] else "unknown_task"

        cf_index = next((i for i, token in enumerate(parts) if token.startswith("cf")), None)
        cf_dir_name = parts[cf_index] if cf_index is not None else "cfUnknown"

        end_idx = cf_index if cf_index is not None else len(parts)
        metric_tokens = []
        for token in reversed(parts[2:end_idx]):
            if metric_token_pattern.fullmatch(token):
                metric_tokens.append(token)
            else:
                if metric_tokens:
                    break

        metric = "_".join(reversed(metric_tokens)) if metric_tokens else None
        metric_suffix = f"_{metric}" if metric else ""
        dest_name = f"paths{metric_suffix}.pkl"

        dest_dir = Path("./data") / model / task / cf_dir_name
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"❌ Could not create destination folder {dest_dir}: {e}")
            continue

        dest_path = dest_dir / dest_name
        try:
            shutil.copy2(src, dest_path)
            copied += 1
            print(f"✅ Copied: {src.name} -> {dest_path}")
        except Exception as e:
            print(f"❌ Failed to copy {src} -> {dest_path}: {e}")

    print(f"\nDone. Files copied: {copied}")
    
if __name__ == "__main__":
    copy_path_from_folder("../experiments/website_data/saved_paths/")