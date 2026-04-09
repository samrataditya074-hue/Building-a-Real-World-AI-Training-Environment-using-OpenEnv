"""Verification script: confirms all graders return scores strictly in (0, 1)."""
import sys
sys.path.insert(0, '.')

from graders import GRADERS, _clamp_score, grade_easy, grade_medium, grade_hard

print("=== GRADERS Registry ===")
print("Keys:", list(GRADERS.keys()))
print()

print("=== Score Boundary Tests ===")
print("_clamp_score(0.0) =", _clamp_score(0.0))
print("_clamp_score(0.5) =", _clamp_score(0.5))
print("_clamp_score(1.0) =", _clamp_score(1.0))
print()

print("=== Empty History Tests ===")
print("grade_easy([]) =", grade_easy([]))
print("grade_medium([]) =", grade_medium([]))
print("grade_hard([]) =", grade_hard([]))
print()

step = {
    'Quarter': 1, 'Cash': 200000, 'Profit': 1000,
    'Valuation': 250000, 'Our_Price': 40,
    'Competitor_Price': 50, 'Morale': 80
}
history4 = [step] * 4
history50 = [step] * 50

print("=== Full History Tests ===")
print("grade_easy(4 quarters) =", grade_easy(history4))
print("grade_medium(50 quarters) =", grade_medium(history50))
print("grade_hard(50 quarters) =", grade_hard(history50))
print()

print("=== ALL SCORES STRICTLY IN (0, 1)? ===")
all_ok = True
for name in ['easy', 'medium', 'hard']:
    fn = GRADERS[name]
    for label, h in [("empty", []), ("4q", history4), ("50q", history50)]:
        s = fn(h)
        status = "OK" if 0.0 < s < 1.0 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {name}/{label}: {s:.4f} [{status}]")

print()
if all_ok:
    print("ALL PASS! Every score is strictly between 0 and 1.")
else:
    print("SOME FAILED! Check output above.")
