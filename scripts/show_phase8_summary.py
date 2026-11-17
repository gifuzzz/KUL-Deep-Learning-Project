#!/usr/bin/env python3
import json

with open("models/phase8_evaluations/ablation_summary.json") as f:
    data = json.load(f)

models = list(data.keys())
pipe = "|"

print("="*96)
print(f"{pipe} Model{' '*26} {pipe} Obs Cov {pipe} Vis Cov {pipe} Wait % {pipe} Osc %  {pipe} Rooms {pipe}")
print("="*96)
for m in models:
    e = data[m]["exploration_benchmark"]
    obs = e["observed_coverage_mean"]
    vis = e["visited_coverage_mean"]
    wait = e["wait_action_fraction_mean"]
    osc = e["oscillation_percentage_mean"]
    rms = e["unique_rooms_entered_mean"]
    print(f"{pipe} {m:<31} {pipe} {obs:6.2f}% {pipe} {vis:6.2f}% {pipe} {wait:5.1f}% {pipe} {osc:5.1f}% {pipe} {rms:5.2f} {pipe}")
print("="*96)

print("\n" + "="*102)
print(f"{pipe} Model{' '*26} {pipe} Cov@50 {pipe} Cov@100 {pipe} Cov@200 {pipe} Cov@384 {pipe} Delta(50->384) {pipe}")
print("="*102)
for m in models:
    e = data[m]["exploration_benchmark"]
    c50 = e.get("coverage_at_50_mean", 0.0)
    c100 = e.get("coverage_at_100_mean", 0.0)
    c200 = e.get("coverage_at_200_mean", 0.0)
    c384 = e.get("coverage_at_384_mean", 0.0)
    delta = c384 - c50
    print(f"{pipe} {m:<31} {pipe} {c50:5.2f}% {pipe} {c100:6.2f}% {pipe} {c200:6.2f}% {pipe} {c384:6.2f}% {pipe} {delta:+13.2f}% {pipe}")
print("="*102)

print("\n" + "="*96)
print(f"{pipe} Model{' '*26} {pipe} Sp 2-6 {pipe} Sp 7-12 {pipe} Sp 13-20 {pipe} Sp 21-35 {pipe} Sp Dist {pipe}")
print("="*96)
for m in models:
    s = data[m]["spawn_ladder"]
    sp2 = s["spawn_2_6"]["rescue_rate"]
    sp7 = s["spawn_7_12"]["rescue_rate"]
    sp13 = s["spawn_13_20"]["rescue_rate"]
    sp21 = s["spawn_21_35"]["rescue_rate"]
    spdist = s["spawn_distant"]["rescue_rate"]
    print(f"{pipe} {m:<31} {pipe} {sp2:5.1f}% {pipe} {sp7:6.1f}% {pipe} {sp13:7.1f}% {pipe} {sp21:7.1f}% {pipe} {spdist:6.1f}% {pipe}")
print("="*96)

print("\n" + "="*106)
print(f"{pipe} Model{' '*26} {pipe} Near   {pipe} Medium {pipe} Brg A  {pipe} Brg B  {pipe} Brg C  {pipe} Distant {pipe}")
print("="*106)
for m in models:
    x = data[m]["exit_ladder"]
    nr = x["near"]["rescue_rate"]
    md = x["medium"]["rescue_rate"]
    ba = x["bridge_a"]["rescue_rate"]
    bb = x["bridge_b"]["rescue_rate"]
    bc = x["bridge_c"]["rescue_rate"]
    dst = x["distant"]["rescue_rate"]
    print(f"{pipe} {m:<31} {pipe} {nr:5.1f}% {pipe} {md:5.1f}% {pipe} {ba:5.1f}% {pipe} {bb:5.1f}% {pipe} {bc:5.1f}% {pipe} {dst:6.1f}% {pipe}")
print("="*106)

print("\n" + "="*106)
print(f"{pipe} Model{' '*26} {pipe} Sp 7-12 Disc {pipe} Sp 7-12 RetCond {pipe} Sp 13-20 Disc {pipe} Sp 13-20 Ret {pipe}")
print("="*106)
for m in models:
    s = data[m]["spawn_ladder"]
    d7 = s["spawn_7_12"]["discovery_rate"]
    r7 = s["spawn_7_12"]["return_conditional"]
    d13 = s["spawn_13_20"]["discovery_rate"]
    r13 = s["spawn_13_20"]["return_conditional"]
    print(f"{pipe} {m:<31} {pipe} {d7:11.1f}% {pipe} {r7:14.1f}% {pipe} {d13:12.1f}% {pipe} {r13:9.1f}% {pipe}")
print("="*106)
