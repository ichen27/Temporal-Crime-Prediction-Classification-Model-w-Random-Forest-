"""Fetch the existing historical input from an immutable repository revision."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

SOURCE_URL = ('https://raw.githubusercontent.com/ichen27/Syracuse-Crime-Data-Analysis/'
              '5999b5c351c1819541b94d860ec4373ae0cdb29b/crime_data.db')
EXPECTED_SHA256 = '1ca006988625cddc35f03c0c64154a817e357d9edcee6792a4eea2e8507046d9'


def download(destination):
    destination = Path(destination)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == EXPECTED_SHA256:
            return destination
        raise ValueError(f'{destination} already exists with different content; choose another path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'syracuse-temporal-model/0.2'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(10_000_001)
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Downloaded data failed its expected SHA-256 verification')
    # Publish atomically without overwriting a file created concurrently.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('work/crime_data.db'))
    args = parser.parse_args()
    try:
        print(download(args.output))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
