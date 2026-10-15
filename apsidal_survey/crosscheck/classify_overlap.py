import re
from collections import defaultdict

src = open("group_statistics.py").read()
sets = {}
for name, body in re.findall(r"^([A-Z_]+)\s*=\s*\{(.*?)\}", src, re.S | re.M):
    sets[name] = {int(x) for x in re.findall(r"\b\d{5,10}\b", body)}

print("sets found:", {n: len(s) for n, s in sets.items()})

overlap = [int(x) for x in open("overlap_175.txt").read().split()]
groups = defaultdict(list)
for t in overlap:
    hits = [n for n, s in sets.items() if t in s]
    groups[", ".join(hits) or "UNLISTED (other)"].append(t)

print(f"\noverlap: {len(overlap)} systems")
for k, v in sorted(groups.items()):
    print(f"\n{k} ({len(v)}):")
    print(" ".join(str(x) for x in sorted(v)))
