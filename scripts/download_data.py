from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factcheck.config import get_settings

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
WANTED = {"corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"}


def main() -> None:
    settings = get_settings()
    target = Path(settings.data_dir)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    written = []
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in WANTED:
                data = archive.extractfile(member).read()
                (target / name).write_bytes(data)
                written.append(name)
    missing = WANTED - set(written)
    if missing:
        raise SystemExit(f"Missing from archive: {sorted(missing)}")
    print(f"Saved {sorted(written)} to {target}")


if __name__ == "__main__":
    main()
