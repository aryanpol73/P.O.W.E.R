"""Escape '|' inside inline-code spans of Markdown table rows so tables don't split.

Usage: python scripts/report/escape_pipes.py docs/*.md
"""

import re
import sys
from pathlib import Path

PIPE = re.compile(r"(?<!\\)\|")
CODE = re.compile(r"`[^`]*`")

for f in sys.argv[1:]:
    path = Path(f)
    lines = path.read_text(encoding="utf-8").split("\n")
    n = 0
    out = []
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("|"):
            new = CODE.sub(lambda m: PIPE.sub(lambda _: "\\|", m.group(0)), line)
            if new != line:
                n += 1
            line = new
        out.append(line)
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f, n, "rows fixed")
