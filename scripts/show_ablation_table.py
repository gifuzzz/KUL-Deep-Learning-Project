import json

with open("models/architecture_ablation/pilot_summary.json") as f:
    data = json.load(f)

params_map = {"baseline": "1,431,168", "deep128": "2,761,600", "deep256": "5,782,656"}
inits = {"baseline": "Continuation", "deep128": "Scratch", "deep256": "Scratch"}

print("=" * 115)
print("PILOT ABLATION COMPARISON (50,000 Environment Steps on Stage 2C)")
print("=" * 115)
header = "%-12s | %-12s | %10s | %8s | %9s | %9s | %9s | %10s | %10s | %8s" % (
    "Model", "Init", "Params", "Steps/s", "Near Det%", "d=6 Det%", "d=8 Det%", "d=10 Det%", "d=12 Det%", "Distant%"
)
print(header)
print("-" * 115)

for arch in ["baseline", "deep128", "deep256"]:
    d = data[arch]
    det = {tier: vals.get("deterministic", {}) for tier, vals in d["eval_metrics"].items()}
    
    near = "%.1f%%" % det.get("Near (d=2, steps=64)", {}).get("rescue_success", 0.0)
    m6 = "%.1f%%" % det.get("Medium (d=6, steps=128)", {}).get("rescue_success", 0.0)
    b8 = "%.1f%%" % det.get("Bridge A (d=8, steps=160)", {}).get("rescue_success", 0.0)
    b10 = "%.1f%%" % det.get("Bridge B (d=10, steps=192)", {}).get("rescue_success", 0.0)
    b12 = "%.1f%%" % det.get("Bridge C (d=12, steps=256)", {}).get("rescue_success", 0.0)
    dist = "%.1f%%" % det.get("Distant (d=None, steps=384)", {}).get("rescue_success", 0.0)
    sps = "%.1f" % d.get("steps_per_second", 0.0)
    
    row = "%-12s | %-12s | %10s | %8s | %9s | %9s | %9s | %10s | %10s | %8s" % (
        arch, inits[arch], params_map[arch], sps, near, m6, b8, b10, b12, dist
    )
    print(row)

print("=" * 115)

print("\nSTOCHASTIC MODE EVALUATION (Sampled Actions):")
print("-" * 115)
header_stoch = "%-12s | %-12s | %9s | %9s | %9s | %10s | %10s | %8s" % (
    "Model", "Init", "Near Stoch%", "d=6 Stoch%", "d=8 Stoch%", "d=10 Stoch%", "d=12 Stoch%", "Distant%"
)
print(header_stoch)
print("-" * 115)
for arch in ["baseline", "deep128", "deep256"]:
    d = data[arch]
    stoch = {tier: vals.get("stochastic", {}) for tier, vals in d["eval_metrics"].items()}
    near = "%.1f%%" % stoch.get("Near (d=2, steps=64)", {}).get("rescue_success", 0.0)
    m6 = "%.1f%%" % stoch.get("Medium (d=6, steps=128)", {}).get("rescue_success", 0.0)
    b8 = "%.1f%%" % stoch.get("Bridge A (d=8, steps=160)", {}).get("rescue_success", 0.0)
    b10 = "%.1f%%" % stoch.get("Bridge B (d=10, steps=192)", {}).get("rescue_success", 0.0)
    b12 = "%.1f%%" % stoch.get("Bridge C (d=12, steps=256)", {}).get("rescue_success", 0.0)
    dist = "%.1f%%" % stoch.get("Distant (d=None, steps=384)", {}).get("rescue_success", 0.0)
    row = "%-12s | %-12s | %9s | %9s | %9s | %10s | %10s | %8s" % (
        arch, inits[arch], near, m6, b8, b10, b12, dist
    )
    print(row)
print("=" * 115)
