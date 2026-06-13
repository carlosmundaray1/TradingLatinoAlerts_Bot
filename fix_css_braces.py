#!/usr/bin/env python3
"""Fix CSS curly braces inside f-string template - need to double {{ and }}"""

with open("tradinglatino_hmm_dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the hybrid CSS section
marker = "/* === HYBRID ALERT SYSTEM === */"
idx = content.find(marker)
if idx < 0:
    print("[FAIL] Could not find hybrid CSS section")
    exit(1)

# Find the end of this CSS section (next section or </style>)
end_marker = "</style></style>"
end_idx = content.find(end_marker, idx)
if end_idx < 0:
    print("[FAIL] Could not find end of CSS section")
    exit(1)

# Extract the CSS section
css_section = content[idx:end_idx]

# Double all single braces (but not already doubled ones)
# We need to replace { with {{ and } with }} 
# But we must be careful not to double already doubled braces

fixed_css = css_section.replace("{{", "\x00LB\x00").replace("}}", "\x00RB\x00")
fixed_css = fixed_css.replace("{", "{{").replace("}", "}}")
fixed_css = fixed_css.replace("\x00LB\x00", "{{").replace("\x00RB\x00", "}}")

# Replace in content
content = content[:idx] + fixed_css + content[end_idx:]

with open("tradinglatino_hmm_dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"[OK] Fixed CSS curly braces (section from offset {idx} to {end_idx})")
print(f"[OK] File saved")
