import numpy as np
import matplotlib.pyplot as plt


x_data = [0, 10, 20, 35, 50, 65, 80, 90, 100]
y_data = [25.0, 26.7, 29.4, 33.2, 35.5, 36.1, 37.8, 38.9, 40.0]
x_target = 45  

def lagrange_interpolation(x_points, y_points, x_value):
    """Compute interpolated value using Lagrange polynomial"""
    n = len(x_points)
    result = 0.0
    for i in range(n):
        Li = 1.0
        for j in range(n):
            if i != j:
                Li *= (x_value - x_points[j]) / (x_points[i] - x_points[j])
        result += y_points[i] * Li
    return result

def lagrange_basis_at_value(x_points, i, x_value):
    """Calculate Li(x_value) for basis function i"""
    Li = 1.0
    for j in range(len(x_points)):
        if i != j:
            Li *= (x_value - x_points[j]) / (x_points[i] - x_points[j])
    return Li


def divided_diff_table(x_points, y_points):
    """Build divided difference table"""
    n = len(x_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_points[i + j] - x_points[i])
    return table

def newton_interpolation(x_points, diff_table, x_value):
    """Compute interpolated value using Newton's method"""
    n = len(x_points)
    result = diff_table[0, 0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x_value - x_points[i - 1])
        result += diff_table[0, i] * product_term
    return result

def select_nearest_nodes(x_data, y_data, x_target, degree):
    """Select k = degree + 1 nearest nodes to x_target"""
    k = degree + 1
    idx_sorted = np.argsort(np.abs(np.array(x_data) - x_target))
    chosen_idx = sorted(idx_sorted[:k])
    x_nodes = [x_data[i] for i in chosen_idx]
    y_nodes = [y_data[i] for i in chosen_idx]
    return x_nodes, y_nodes


lagrange_results = []
newton_results = []
previous_lag = None
previous_new = None

degrees_to_compute = [2, 3, 4, 8]

for degree in degrees_to_compute:
    sub_x, sub_y = select_nearest_nodes(x_data, y_data, x_target, degree)
    
    lag_value = lagrange_interpolation(sub_x, sub_y, x_target)
    lag_delta = abs(lag_value - previous_lag) if previous_lag is not None else None
    lagrange_results.append((degree, lag_value, lag_delta, sub_x, sub_y))
    previous_lag = lag_value
    
    diff_table = divided_diff_table(sub_x, sub_y)
    new_value = newton_interpolation(sub_x, diff_table, x_target)
    new_delta = abs(new_value - previous_new) if previous_new is not None else None
    newton_results.append((degree, new_value, new_delta, sub_x, diff_table))
    previous_new = new_value



print("=" * 80)
print("TEMPERATURE INTERPOLATION AT x = 45 km")
print("=" * 80)

print("\n" + "─" * 80)
print("LAGRANGE INTERPOLATION RESULTS")
print("─" * 80)

for deg, val, delta, nodes, y_nodes in lagrange_results:
    print(f"\n{'='*70}")
    print(f"Degree {deg} Lagrange Polynomial")
    print(f"{'='*70}")
    print(f"Nodes (km):  {nodes}")
    print(f"Temps (°C):  {y_nodes}")
    
    print(f"\nBasis function values at x = {x_target} km:")
    for i in range(len(nodes)):
        Li_value = lagrange_basis_at_value(nodes, i, x_target)
        print(f"  L{i}({x_target}) = {Li_value:.8f}")
    
    print(f"\nEvaluation:")
    eval_terms = []
    for i in range(len(nodes)):
        Li_value = lagrange_basis_at_value(nodes, i, x_target)
        eval_terms.append(f"{Li_value:.6f} × {y_nodes[i]:.1f}")
    print(f"  P{deg}({x_target}) = {' + '.join(eval_terms)}")
    print(f"  P{deg}({x_target}) = {val:.6f} °C")
    
    if delta is not None:
        print(f"\n  Δ{deg} = |P{deg}({x_target}) - P{deg-1}({x_target})| = {delta:.6f} °C")

print("\n" + "─" * 80)
print("NEWTON'S DIVIDED DIFFERENCE RESULTS")
print("─" * 80)

for deg, val, delta, nodes, table in newton_results:
    print(f"\n{'='*70}")
    print(f"Degree {deg} Newton Polynomial")
    print(f"{'='*70}")
    print(f"Nodes (km): {nodes}")
    
    print("\nDivided Difference Table:")
    n = len(nodes)
    print(f"{'i':<4} {'x_i (km)':<12} {'f[x_i]':<12}", end="")
    for j in range(2, n + 1):
        print(f" {'f[...]_' + str(j):<12}", end="")
    print()
    print("-" * (28 + 13 * (n - 1)))
    
    for i in range(n):
        print(f"{i:<4} {nodes[i]:<12.1f} ", end="")
        for j in range(n):
            if j < n - i:
                print(f"{table[i][j]:<12.6f} ", end="")
            else:
                print(f"{'':12} ", end="")
        print()
    
    print(f"\nN{deg}({x_target}) = {val:.6f} °C")
    
    if delta is not None:
        print(f"Δ{deg} = |N{deg}({x_target}) - N{deg-1}({x_target})| = {delta:.6f} °C")

print("\n" + "=" * 80)
print("COMPARISON OF METHODS")
print("=" * 80)
print(f"{'Degree':<8} {'Lagrange T(45)':<18} {'Newton T(45)':<18} {'Difference':<12}")
print("-" * 80)
for i, deg in enumerate(degrees_to_compute):
    lag_val = lagrange_results[i][1]
    new_val = newton_results[i][1]
    diff = abs(lag_val - new_val)
    print(f"{deg:<8} {lag_val:<18.6f} {new_val:<18.6f} {diff:<12.2e}")


x_vals = np.linspace(0, 100, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(x_data, y_data, color="red", s=80, label="Sensor Data", zorder=5, edgecolors='black')
ax1.axvline(x_target, color="gray", linestyle="--", linewidth=1.5, label=f"x = {x_target} km", alpha=0.7)

for deg, val, delta, nodes, y_nodes in lagrange_results:
    y_vals = [lagrange_interpolation(nodes, y_nodes, xv) for xv in x_vals]
    ax1.plot(x_vals, y_vals, label=f"P{deg}(x) - Degree {deg}", linewidth=2)

ax1.set_title("Lagrange Interpolation", fontsize=14, fontweight='bold')
ax1.set_xlabel("Distance (km)", fontsize=12)
ax1.set_ylabel("Temperature (°C)", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.scatter(x_data, y_data, color="red", s=80, label="Sensor Data", zorder=5, edgecolors='black')
ax2.axvline(x_target, color="gray", linestyle="--", linewidth=1.5, label=f"x = {x_target} km", alpha=0.7)

for deg, val, delta, nodes, table in newton_results:
    y_vals = [newton_interpolation(nodes, table, xv) for xv in x_vals]
    ax2.plot(x_vals, y_vals, label=f"N{deg}(x) - Degree {deg}", linewidth=2)

ax2.set_title("Newton's Divided Difference Interpolation", fontsize=14, fontweight='bold')
ax2.set_xlabel("Distance (km)", fontsize=12)
ax2.set_ylabel("Temperature (°C)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interpolation_curves.png', dpi=300, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

lag_degrees = [deg for deg, _, delta, _, _ in lagrange_results if delta is not None]
lag_deltas = [delta for _, _, delta, _, _ in lagrange_results if delta is not None]

ax1.plot(lag_degrees, lag_deltas, marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
ax1.set_title("Lagrange Interpolation Convergence", fontsize=14, fontweight='bold')
ax1.set_xlabel("Polynomial Degree (k)", fontsize=12)
ax1.set_ylabel("Δₖ = |Pₖ(45) − Pₖ₋₁(45)| (°C)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

new_degrees = [deg for deg, _, delta, _, _ in newton_results if delta is not None]
new_deltas = [delta for _, _, delta, _, _ in newton_results if delta is not None]

ax2.plot(new_degrees, new_deltas, marker='s', linewidth=2.5, markersize=8, color='#A23B72')
ax2.set_title("Newton's Method Convergence", fontsize=14, fontweight='bold')
ax2.set_xlabel("Polynomial Degree (k)", fontsize=12)
ax2.set_ylabel("Δₖ = |Nₖ(45) − Nₖ₋₁(45)| (°C)", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('convergence_curves.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nEstimated temperature at x = 45 km:")
print(f"  Lagrange (Degree 8):  {lagrange_results[-1][1]:.4f} °C")
print(f"  Newton (Degree 8):    {newton_results[-1][1]:.4f} °C")
print(f"  Difference:           {abs(lagrange_results[-1][1] - newton_results[-1][1]):.2e} °C")
