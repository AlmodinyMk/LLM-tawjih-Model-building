<!-- tawjih 51c684 machine: الحل بالمميز -->
=== instructions ===

=== code ===
import json, re, math, cmath
k = json.loads(re.search(r"\{.*\}", input, re.S).group(0))
a, b, c = k["a"], k["b"], k["c"]
if a == 0:
    raise ValueError("a = 0: not quadratic")
d = b*b - 4*a*c
root = math.sqrt(d) if d >= 0 else cmath.sqrt(d)
r = [(-b + root) / (2*a), (-b - root) / (2*a)]
vars.update(a=a, b=b, c=c, d=d, roots=r)
output = json.dumps({"a": a, "b": b, "c": c, "delta": d, "roots": [str(x) for x in r]})
