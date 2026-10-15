import re
src = open("group_statistics.py").read()
analyzed = {int(x) for x in open("victor_analyzed.txt").read().split()} - {408651399}
sets = {}
for name, body in re.findall(r"^([A-Z_]+)\s*=\s*\{(.*?)\}", src, re.S | re.M):
    sets[name] = {int(x) for x in re.findall(r"\b\d{5,10}\b", body)}

print(f"analyzed: {len(analyzed)}\n")
for n, s in sets.items():
    if s:
        print(f"{n}: {len(s)} listed, {len(s & analyzed)} analyzed; "
              f"not analyzed: {sorted(s - analyzed)}")

print("\nTICs in more than one set:")
names = [n for n in sets if sets[n] and n != "DUPLICATE"]
for i, a in enumerate(names):
    for b in names[i+1:]:
        both = sets[a] & sets[b]
        if both:
            print(f"  {a} & {b}: {sorted(both)}")

for n in ("PRECESSION_LTTE", "TENTATIVE"):
    m = re.search(rf"^{n}\s*=\s*\{{.*?\}}", src, re.S | re.M)
    print(f"\n{m.group(0)}")
