f = open("2db.cfg")
lines = []
for line in f:
    lines.append(line.strip())
print(len(set(lines)))
