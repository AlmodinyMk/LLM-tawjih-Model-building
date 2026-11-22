<!-- tawjih de5595 machine: التحقق العددي -->
=== instructions ===

=== code ===
a, b, c = vars["a"], vars["b"], vars["c"]
res = [abs(a*x*x + b*x + c) for x in vars["roots"]]
ok = all(e < 1e-9 for e in res)
output = input + "\nverification by substitution: %s (residuals %s)" % ("PASS" if ok else "FAIL", res)
