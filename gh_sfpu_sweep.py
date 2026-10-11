import os
import subprocess
import json
import time

directory = "/home/abhishek/Ventures/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
files = os.listdir(directory)
ops = [f.replace("ckernel_sfpu_", "").replace(".h", "") for f in files if f.startswith("ckernel_sfpu_") and f.endswith(".h")]

# Exclude some obvious non-unary math ops
exclude = ["binary", "add", "sub", "mul", "div", "redu", "comp", "conversions", "typecast", "copy", "mac", "reshuffle", "tiled", "quant", "bitwise", "shift", "max", "min", "mask", "sum", "sign", "isclose"]
ops = [op for op in ops if not any(ex in op for ex in exclude)]

print(f"Scanning {len(ops)} ops...")

results = {}
for op in ops:
    query = f"in:title,body {op}"
    try:
        # Search issues in tenstorrent/tt-metal
        cmd = f"gh issue list --repo tenstorrent/tt-metal --search '{query}' --state all --json number"
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        data = json.loads(out)
        
        # Search PRs in tenstorrent/tt-metal
        cmd_pr = f"gh pr list --repo tenstorrent/tt-metal --search '{query}' --state all --json number"
        out_pr = subprocess.check_output(cmd_pr, shell=True, text=True, stderr=subprocess.DEVNULL)
        data_pr = json.loads(out_pr)
        
        total = len(data) + len(data_pr)
        results[op] = total
        print(f"Op: {op:<15} -> {total} mentions")
    except Exception as e:
        print(f"Failed {op}: {e}")
    time.sleep(0.5)

untouched = [op for op, count in results.items() if count == 0]
print(f"\nUntouched ops ({len(untouched)}):")
for op in untouched:
    print(f" - {op}")
