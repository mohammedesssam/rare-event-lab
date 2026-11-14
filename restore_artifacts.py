"""Restore supplementary original project package. Standard library only.
Does not install packages, load models, execute project code, or overwrite files.
The repository's ordinary source files are also available for inspection.
"""
import base64, hashlib, io, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PARTS = 4
SHA256 = '5c6a6ac890ea774a06ecee67a57ccd48d9a003b453115955bb9b364b5e6234ec'

def main():
    encoded = ''.join((ROOT / f'project.zip.b64.part{i}').read_text() for i in range(1, PARTS + 1))
    raw = base64.b64decode(encoded)
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit('Archive checksum mismatch')
    target = ROOT / 'restored'
    if target.exists():
        raise SystemExit('restored already exists; move or remove it before restoring again')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for item in archive.infolist():
            if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts:
                raise SystemExit('Unsafe archive member')
        archive.extractall(target)
    print('Restored verified project to', target / 'rare-event-lab')
    print('Read that directory README before installing or running anything.')

if __name__ == '__main__':
    main()
