import io
import re
import sys

PIPE = re.compile(r"(?<!\\)\|")
CODE = re.compile(r"`[^`]*`")

for f in sys.argv[1:]:
    lines = io.open(f, encoding="utf-8").read().split("\n")
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
    io.open(f, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(f, n, "rows fixed")
