with open("tradinglatino_hmm_dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    for j, ch in enumerate(line):
        if ch == "\\" and j + 1 < len(line) and line[j+1] == '"':
            print(f"L{i+1} col{j}: {repr(line.rstrip()[:150])}")
    
print("--- Checking for backslash in f-string expressions ---")
for i, line in enumerate(lines):
    stripped = line.rstrip()
    if "f" in stripped and "{" in stripped:
        # Check if there's an escaped quote inside expression
        in_brace = False
        braces_content = ""
        for ch in stripped:
            if ch == "{":
                in_brace = True
                braces_content = ""
            elif ch == "}":
                if "\\" in braces_content:
                    print(f"L{i+1}: BACKSLASH IN F-STRING EXPR: {stripped[:120]}")
                in_brace = False
            elif in_brace:
                braces_content += ch

print("Done")
