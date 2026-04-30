#!/usr/bin/env python3
"""
Migration script for prepare_*.py files to use cli_entrypoint.

Run: python experiments/migrate_prepare_files.py
"""

import re
from pathlib import Path


def migrate_file(f: Path) -> bool:
    """Migrate a single prepare file."""
    content = f.read_text()

    if "cli_entrypoint" in content:
        return False  # Already migrated

    exp_name = f.stem.replace("prepare_", "").replace("_", " ").title()

    # Add import
    if "import numpy as np" in content:
        content = content.replace(
            "import numpy as np\n",
            "import numpy as np\n\nfrom apgi_cli import cli_entrypoint, create_standard_parser\n",
        )

    # Update verify signature
    content = re.sub(
        r"def verify\(\)(?:\s*->\s*None)?\s*:",
        'def verify() -> int:\n    """Verify configuration and return status."""',
        content,
    )

    # Find and update the end of verify function
    lines = content.split("\n")
    new_lines = []
    in_verify = False
    verify_start = -1

    for i, line in enumerate(lines):
        if "def verify()" in line:
            in_verify = True
            verify_start = i
        elif in_verify and line.startswith("def ") and "verify" not in line:
            in_verify = False
        elif in_verify and line.strip().startswith("if __name__"):
            in_verify = False

        new_lines.append(line)

        # Add return 0 before the function ends
        if in_verify and i > verify_start + 1:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if (
                    next_line.strip().startswith("if __name__")
                    or next_line.strip() == ""
                ):
                    if "return 0" not in line and not any(
                        prev_line.strip() == "return 0"
                        for prev_line in lines[i - 2 : i]
                    ):
                        new_lines.append("    return 0")
                        in_verify = False

    content = "\n".join(new_lines)

    # Add main() and update entry point
    main_func = f'''def main() -> int:
    """Entry point for {exp_name} preparation."""
    return verify()

'''

    content = re.sub(
        r'if __name__ == ["\']__main__["\']:\s*\n\s*verify\(\)',
        main_func + f"""if __name__ == "__main__":
    parser = create_standard_parser("Prepare {exp_name} experiment")
    cli_entrypoint(main, parser)""",
        content,
    )

    f.write_text(content)
    return True


def main() -> int:
    experiments_dir = Path(__file__).parent
    migrated = 0
    skipped = 0

    for f in sorted(experiments_dir.glob("prepare_*.py")):
        if migrate_file(f):
            print(f"Migrated: {f.name}")
            migrated += 1
        else:
            print(f"Skipped: {f.name}")
            skipped += 1

    print(f"\nTotal: {migrated} migrated, {skipped} already done")
    return 0


if __name__ == "__main__":
    main()
