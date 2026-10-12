"""Build a fresh public source directory/ZIP from an explicit allowlist."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {'__pycache__', '.venv', '.local', '.git', 'work', 'release'}
TEXT_SUFFIXES = {'.py', '.md', '.json', '.html', '.js', '.css', '.txt', '.cmd'}


def public_file(path):
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(p in BLOCKED_DIRS for p in parts):
        return False
    if path.suffix.lower() in {'.pdf', '.pyc', '.pyo'}:
        return False
    if path.name.startswith('.env') and path.name != '.env.example':
        return False
    if parts[:3] in [('docs', 'theory', 'course'), ('docs', 'theory', 'textbook')]:
        return False
    if parts[:3] == ('docs', 'theory', 'example_i12') and path.suffix.lower() == '.png':
        return False
    if rel.as_posix() == 'examples/textbook_ii2_question.png':
        return False
    return True


def selected_files():
    selected = []
    for name in ['src', 'tests', 'examples', 'docs', 'tools']:
        selected.extend(p for p in (ROOT / name).rglob('*') if p.is_file() and public_file(p))
    selected.extend(p for p in ROOT.iterdir() if p.is_file() and public_file(p)
                    and (p.suffix in {'.py', '.md', '.cmd'} or p.name in {
                        'LICENSE', 'requirements.txt', '.gitignore', '.gitattributes', '.env.example'}))
    selected.extend(p for p in [ROOT/'results/.gitkeep', ROOT/'data/.gitkeep'] if p.exists())
    return sorted(set(selected))


def scan_credentials(files):
    # Reject recognizable secret forms without printing the matched secret.
    patterns = [r'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}',
                r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    failures = []
    for path in files:
        if path.suffix not in TEXT_SUFFIXES and path.name != '.env.example':
            continue
        if path.name.endswith('.min.js'):
            continue  # Vendored library; preserve rather than rewrite its contents.
        content = path.read_text(encoding='utf-8-sig')
        if any(re.search(pattern, content) for pattern in patterns):
            failures.append(path.relative_to(ROOT).as_posix())
    if failures:
        raise RuntimeError('Potential credentials in selected files: ' + ', '.join(failures))


def main():
    files = selected_files()
    scan_credentials(files)
    tag = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    base = ROOT / 'release' / tag
    target = base / 'space-frame-agent'
    target.mkdir(parents=True, exist_ok=False)
    manifest = []
    for source in files:
        rel = source.relative_to(ROOT)
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append({'path':rel.as_posix(), 'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    (base/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = base / 'space-frame-agent-source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(target.rglob('*')):
            if path.is_file():
                out.write(path, path.relative_to(base).as_posix())
    print(json.dumps({'directory':str(target), 'zip':str(archive),
                      'file_count':len(manifest), 'zip_bytes':archive.stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
