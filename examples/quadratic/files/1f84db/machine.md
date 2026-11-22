<!-- tawjih 1f84db machine: التحقق بمجموع وجداء الجذرين -->
=== instructions ===

=== code ===
a, b, c = vars["a"], vars["b"], vars["c"]
r1, r2 = vars["roots"]
ok = abs((r1 + r2) + b / a) < 1e-9 and abs(r1 * r2 - c / a) < 1e-9
output = "verification by sum and product: %s (sum %s, product %s)" % (
    "PASS" if ok else "FAIL", r1 + r2, r1 * r2)
