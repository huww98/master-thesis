import sys
import re

all_txt = sys.stdin.read()
items = re.split(r'\\bibitem\[.+?\]\{.+?\}', all_txt, flags=re.DOTALL)[1:]

for i, txt in enumerate(items):
    lines = [l.strip() for l in txt.strip().splitlines()]
    if i == len(items) - 1:
        lines = lines[:-1]
    txt = ' '.join(lines)
    sys.stdout.write(f'[{i + 1}] {txt}\n')
