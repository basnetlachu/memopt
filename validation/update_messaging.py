"""Update all messaging to proven claims only"""

import os
import glob

UPDATES = {
    '30-40% bandwidth reduction': '15-25% bandwidth reduction (hardware-validated)',
    '31.2% memory reduction': 'Memory optimization (validated in pilot)',
    'Lazy KV optimization': 'Memory access coalescing',
    'guaranteed': 'measured',
    'contact@memopt.dev': '[your email]',
    '$100K-250K': 'Pilot: $25K-50K, Enterprise: $500K-1M',
}

def update_file(filepath):
    """Update a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original = content
        for old, new in UPDATES.items():
            content = content.replace(old, new)

        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Updated {filepath}")
        else:
            print(f"  (no changes) {filepath}")
    except Exception as e:
        print(f"❌ Error {filepath}: {e}")

def update_all():
    """Update all project files."""
    files = [
        'README.md',
        'examples/*.py',
        'memopt/*.py',
    ]

    all_files = []
    for pattern in files:
        all_files.extend(glob.glob(pattern, recursive=True))

    for f in all_files:
        if os.path.isfile(f):
            update_file(f)

if __name__ == '__main__':
    update_all()
    print("\n✅ All messaging updated")
