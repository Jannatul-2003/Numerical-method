import numpy as np
import matplotlib.pyplot as plt

x_data = [0, 1, 2.5, 3, 4.5, 5, 6]
y_data = [2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000]
x_target = 3.5

def lagrange_interpolation(x_points, y_points, x_value):
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
    value = lagrange_interpolation(sub_x, sub_y, x_target)
    delta = abs(value - previous_val) if previous_val is not None else None
    results.append((degree, value, delta, sub_x, sub_y))
    previous_val = value

for deg, val, delta, nodes, y_nodes in results:
    print(f"\n{'='*60}")
    print(f"=== Degree {deg} Lagrange Interpolant ===")
    print(f"{'='*60}")
    print(f"Nodes used: {nodes}")
    print(f"f(x) values: {y_nodes}")
    
    print(f"\nLagrange Basis Functions:")
    print("-" * 60)
    n = len(nodes)
    for i in range(n):
        terms = []
        for j in range(n):
            if i != j:
                terms.append(f"(x - {nodes[j]:.4g})")
        numerator = " × ".join(terms)
        
        denom_terms = []
        for j in range(n):
            if i != j:
                denom_terms.append(f"({nodes[i]:.4g} - {nodes[j]:.4g})")
        denominator = " × ".join(denom_terms)
        
        print(f"L{i}(x) = {numerator}")
        print(f"        {'-' * len(numerator)}")
        print(f"        {denominator}")
        print()
    
    print(f"Basis function values at x = {x_target}:")
    print("-" * 60)
    for i in range(n):
        Li_value = lagrange_basis_at_value(nodes, i, x_target)
        print(f"L{i}({x_target}) = {Li_value:.6f}")
    
    print(f"\nGeneral form of P{deg}(x):")
    print("-" * 60)
    general_terms = [f"L{i}(x)·f(x{i})" for i in range(n)]
    print(f"P{deg}(x) = {' + '.join(general_terms)}")
    
    print(f"\nSpecific form with values:")
    print("-" * 60)
    specific_terms = [f"L{i}(x)·{y_nodes[i]:.6f}" for i in range(n)]
    print(f"P{deg}(x) = {' + '.join(specific_terms)}")
    
    print(f"\nEvaluation at x = {x_target}:")
    print("-" * 60)
    eval_terms = []
    for i in range(n):
        Li_value = lagrange_basis_at_value(nodes, i, x_target)
        eval_terms.append(f"({Li_value:.6f})·({y_nodes[i]:.6f})")
    print(f"P{deg}({x_target}) = {' + '.join(eval_terms)}")
    print(f"P{deg}({x_target}) = {val:.6f}")
    
    if delta is not None:
        print(f"\nΔ{deg} = |P{deg}({x_target}) - P{deg-1}({x_target})| = {delta:.6f}")

print("\n" + "="*70)
print("Summary: Convergence of Lagrange Interpolants (nearest nodes)")
print("="*70)
print(f"{'Degree':>6} | {'Nodes Used':<30} | {'P(3.5)':>10} | {'Δk':>10}")
print("-" * 70)
for deg, val, delta, nodes, _ in results:
    node_str = str(nodes)
    delta_str = f"{delta:0.6f}" if delta is not None else "-"
    print(f"{deg:6d} | {node_str:<30} | {val:10.6f} | {delta_str:>10}")

x_vals = np.linspace(min(x_data), max(x_data), 200)

plt.figure(figsize=(9, 6))
plt.scatter(x_data, y_data, color="black", label="Data Points", zorder=5)
plt.axvline(x_target, color="gray", linestyle="--", label="x = 3.5")

for deg, val, delta, nodes, y_nodes in results:
    y_vals = [lagrange_interpolation(nodes, y_nodes, xv) for xv in x_vals]
    plt.plot(x_vals, y_vals, label=f"P{deg}(x)")

plt.title("Lagrange Interpolation (Nearest Nodes Method)")
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
plt.title("Convergence of Lagrange Interpolants")
plt.xlabel("Polynomial Degree (k)")
plt.ylabel("Δk = |Pₖ(3.5) − Pₖ₋₁(3.5)|")
plt.grid(True)
plt.tight_layout()
plt.show()