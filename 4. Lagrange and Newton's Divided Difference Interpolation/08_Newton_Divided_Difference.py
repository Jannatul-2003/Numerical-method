import numpy as np
import matplotlib.pyplot as plt

x_data = [0, 1, 2.5, 3, 4.5, 5, 6]
y_data = [2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000]
x_target = 3.5

def divided_diff_table(x_points, y_points):
    n = len(x_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_points[i + j] - x_points[i])
    return table

def newton_interpolation(x_points, diff_table, x_value):
    n = len(x_points)
    result = diff_table[0, 0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x_value - x_points[i - 1])
        result += diff_table[0, i] * product_term
    return result

def select_nearest_nodes(x_data, y_data, x_target, degree):
    k = degree + 1
    idx_sorted = np.argsort(np.abs(np.array(x_data) - x_target))
    chosen_idx = sorted(idx_sorted[:k])
    x_nodes = [x_data[i] for i in chosen_idx]
    y_nodes = [y_data[i] for i in chosen_idx]
    return x_nodes, y_nodes

results = []
previous_val = None

for degree in range(2, len(x_data)): 
    sub_x, sub_y = select_nearest_nodes(x_data, y_data, x_target, degree)
    diff_table = divided_diff_table(sub_x, sub_y)
    value = newton_interpolation(sub_x, diff_table, x_target)
    delta = abs(value - previous_val) if previous_val is not None else None
    results.append((degree, value, delta, sub_x, diff_table))
    previous_val = value

for deg, val, delta, nodes, table in results:
    print(f"\n=== Degree {deg} Newton Interpolant ===")
    print(f"Nodes used: {nodes}")
    print("\nDivided Difference Table:")
    n = len(nodes)   
    headers = ["x_i", "f[x_i]"] + [f"f[...](order {j})" for j in range(2, n + 1)]
    header = " | ".join(h.center(12) for h in headers[:n+1])
    print(header)
    print("-" * len(header))
    

    for i in range(n):
        row = f"{nodes[i]:<12.4f} | "
        row += " | ".join(f"{table[i][j]:12.6f}" if j < n - i else " " * 12 for j in range(n))
        print(row)
    
    print(f"\nN{deg}(3.5) = {val:.6f}")
    if delta is not None:
        print(f"Δ{deg} = |N{deg} - N{deg-1}| = {delta:.6f}")
    
    print(f"\nSpecific form with values:")
    equation_vals = f"N{deg}(x) = {table[0,0]:.6f}"
    product_str = ""
    for j in range(1, n):
        product_str += f"(x - {nodes[j-1]:.4f})"
        equation_vals += f" + {table[0,j]:.6f}{product_str}"
    print(equation_vals)

print("\nSummary: Convergence of Newton Interpolants (nearest nodes)")
print(f"{'Degree':>6} | {'Nodes Used':<30} | {'N(3.5)':>10} | {'Δk':>10}")
print("-" * 70)
for deg, val, delta, nodes, table in results:
    delta_str = f"{delta:0.6f}" if delta is not None else "-"
    print(f"{deg:6d} | {str(nodes):<30} | {val:10.6f} | {delta_str:>10}")

x_vals = np.linspace(min(x_data), max(x_data), 200)

plt.figure(figsize=(9, 6))
plt.scatter(x_data, y_data, color="black", label="Data Points", zorder=5)
plt.axvline(x_target, color="gray", linestyle="--", label="x = 3.5")

for deg, val, delta, nodes, table in results:
    y_vals = [newton_interpolation(nodes, table, xv) for xv in x_vals]
    plt.plot(x_vals, y_vals, label=f"N{deg}(x)")

plt.title("Newton's Divided Difference Interpolation (Nearest Nodes)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

degrees = [deg for deg, _, _, _, _ in results if deg > 2]  
deltas = [delta for _, _, delta, _, _ in results if delta is not None]

plt.figure(figsize=(8, 5))
plt.plot(degrees, deltas, marker='o', linewidth=2)
plt.title("Convergence of Newton Interpolants")
plt.xlabel("Polynomial Degree (k)")
plt.ylabel("Δk = |Nₖ(3.5) − Nₖ₋₁(3.5)|")
plt.grid(True)
plt.tight_layout()
plt.show()