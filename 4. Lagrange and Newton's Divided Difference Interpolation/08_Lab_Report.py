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

def select_nearest_nodes_balanced(x_data, y_data, x_target, degree):
    k = degree + 1
    x_arr = np.array(x_data)

    center_idx = np.argmin(np.abs(x_arr - x_target))

    left = center_idx
    right = center_idx
    selected = {center_idx}

    while len(selected) < k:
        left_candidate = left - 1
        right_candidate = right + 1

        left_dist = abs(x_arr[left_candidate] - x_target) if left_candidate >= 0 else float('inf')
        right_dist = abs(x_arr[right_candidate] - x_target) if right_candidate < len(x_arr) else float('inf')

        if left_dist <= right_dist:
            if left_candidate >= 0:
                left = left_candidate
                selected.add(left)
            else:
                right = right_candidate
                selected.add(right)
        else:
            if right_candidate < len(x_arr):
                right = right_candidate
                selected.add(right)
            else:
                left = left_candidate
                selected.add(left)

    chosen_idx = sorted(selected)
    x_nodes = [x_data[i] for i in chosen_idx]
    y_nodes = [y_data[i] for i in chosen_idx]
    return x_nodes, y_nodes



lagrange_results = []
newton_results = []
previous_lag = None
previous_new = None

degrees_to_compute = list(range(2, 9))

for degree in degrees_to_compute:
    sub_x, sub_y = select_nearest_nodes_balanced(x_data, y_data, x_target, degree)
    
    lag_value = lagrange_interpolation(sub_x, sub_y, x_target)
    lag_delta = abs(lag_value - previous_lag) if previous_lag is not None else None
    lagrange_results.append((degree, lag_value, lag_delta, sub_x, sub_y))
    previous_lag = lag_value
    
    diff_table = divided_diff_table(sub_x, sub_y)
    new_value = newton_interpolation(sub_x, diff_table, x_target)
    new_delta = abs(new_value - previous_new) if previous_new is not None else None
    newton_results.append((degree, new_value, new_delta, sub_x, diff_table))
    previous_new = new_value


print("\n" + "╔" + "═" * 78 + "╗")
print("║" + " " * 20 + "TEMPERATURE INTERPOLATION AT x = 45 km" + " " * 19 + "║")
print("╚" + "═" * 78 + "╝")

print("\n" + "┌" + "─" * 78 + "┐")
print("│" + " " * 25 + "LAGRANGE INTERPOLATION RESULTS" + " " * 23 + "│")
print("└" + "─" * 78 + "┘")

for deg, val, delta, nodes, y_nodes in lagrange_results:
    print(f"\n{'━' * 80}")
    print(f"  DEGREE {deg} LAGRANGE POLYNOMIAL - P{deg}(x)")
    print(f"{'━' * 80}")
    print(f"  Selected Nodes (km):  {nodes}")
    print(f"  Temperature Data (°C): {y_nodes}")
    print(f"  Number of nodes used:  {len(nodes)}")
    
    print(f"\n  Lagrange Basis Functions at x = {x_target} km:")
    print(f"  {'-' * 76}")
    for i in range(len(nodes)):
        Li_value = lagrange_basis_at_value(nodes, i, x_target)
        print(f"    L_{i}({x_target}) = {Li_value:>12.8f}  [Node at x = {nodes[i]} km, T = {y_nodes[i]}°C]")
    
    print(f"\n  Polynomial Evaluation:")
    print(f"  {'-' * 76}")
    total = 0.0
    for i in range(len(nodes)):
        Li_value = lagrange_basis_at_value(nodes, i, x_target)
        contribution = Li_value * y_nodes[i]
        total += contribution
        print(f"    Term {i+1}: L_{i}({x_target}) × T_{i} = {Li_value:.8f} × {y_nodes[i]:.1f} = {contribution:>10.6f}")
    
    print(f"\n  ╔{'═' * 76}╗")
    print(f"  ║  P_{deg}({x_target}) = {val:>10.6f} °C" + " " * (76 - len(f"  P_{deg}({x_target}) = {val:>10.6f} °C")) + "║")
    print(f"  ╚{'═' * 76}╝")
    
    if delta is not None:
        print(f"\n  Convergence Check:")
        print(f"    Δ_{deg} = |P_{deg}({x_target}) - P_{deg-1}({x_target})| = {delta:.8f} °C")


print("\n\n" + "┌" + "─" * 78 + "┐")
print("│" + " " * 20 + "NEWTON'S DIVIDED DIFFERENCE RESULTS" + " " * 23 + "│")
print("└" + "─" * 78 + "┘")

for deg, val, delta, nodes, table in newton_results:
    print(f"\n{'━' * 80}")
    print(f"  DEGREE {deg} NEWTON POLYNOMIAL - N{deg}(x)")
    print(f"{'━' * 80}")
    print(f"  Selected Nodes (km): {nodes}")
    print(f"  Number of nodes used: {len(nodes)}")
    
    print("\n  Divided Difference Table:")
    print(f"  {'-' * 76}")
    n = len(nodes)
    
    header = f"  {'i':<3} {'x_i':<8} {'f[x_i]':<12}"
    for j in range(2, n + 1):
        header += f"f[..]_{j:<2} "
    print(header)
    print(f"  {'-' * 76}")
    

    for i in range(n):
        row = f"  {i:<3} {nodes[i]:<8.1f} "
        for j in range(n):
            if j < n - i:
                row += f"{table[i][j]:<12.6f} "
        print(row)
    
    print(f"\n  Coefficients for Newton Polynomial:")
    print(f"  {'-' * 76}")
    for i in range(n):
        if i == 0:
            print(f"    a_{i} = f[x_0] = {table[0][i]:.8f}")
        else:
            print(f"    a_{i} = f[x_0, x_1, ..., x_{i}] = {table[0][i]:.8f}")
    
    print(f"\n  ╔{'═' * 76}╗")
    print(f"  ║  N_{deg}({x_target}) = {val:>10.6f} °C" + " " * (76 - len(f"  N_{deg}({x_target}) = {val:>10.6f} °C")) + "║")
    print(f"  ╚{'═' * 76}╝")
    
    if delta is not None:
        print(f"\n  Convergence Check:")
        print(f"    Δ_{deg} = |N_{deg}({x_target}) - N_{deg-1}({x_target})| = {delta:.8f} °C")


print("\n\n" + "╔" + "═" * 78 + "╗")
print("║" + " " * 25 + "COMPARISON OF METHODS" + " " * 32 + "║")
print("╚" + "═" * 78 + "╝\n")

print(f"{'Degree':<8} {'Lagrange P_k(45)':<20} {'Newton N_k(45)':<20} {'|P_k - N_k|':<15}")
print("─" * 80)
for i, deg in enumerate(degrees_to_compute):
    lag_val = lagrange_results[i][1]
    new_val = newton_results[i][1]
    diff = abs(lag_val - new_val)
    print(f"{deg:<8} {lag_val:<20.10f} {new_val:<20.10f} {diff:<15.2e}")


print("\n\n" + "╔" + "═" * 78 + "╗")
print("║" + " " * 28 + "CONVERGENCE SUMMARY" + " " * 31 + "║")
print("╚" + "═" * 78 + "╝\n")

print(f"{'Degree k':<12} {'P_k(45) [°C]':<18} {'Δ_k (Lagrange)':<18} {'N_k(45) [°C]':<18} {'Δ_k (Newton)':<15}")
print("─" * 95)
for i, deg in enumerate(degrees_to_compute):
    lag_val = lagrange_results[i][1]
    lag_delta = lagrange_results[i][2]
    new_val = newton_results[i][1]
    new_delta = newton_results[i][2]
    
    lag_delta_str = f"{lag_delta:.8f}" if lag_delta is not None else "---"
    new_delta_str = f"{new_delta:.8f}" if new_delta is not None else "---"
    
    print(f"{deg:<12} {lag_val:<18.10f} {lag_delta_str:<18} {new_val:<18.10f} {new_delta_str:<15}")


x_vals = np.linspace(0, 100, 300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


ax1.scatter(x_data, y_data, color="red", s=100, label="Sensor Data", zorder=5, 
            edgecolors='black', linewidths=2)
ax1.axvline(x_target, color="gray", linestyle="--", linewidth=2, 
            label=f"x = {x_target} km", alpha=0.7)

colors = plt.cm.viridis(np.linspace(0, 1, len(degrees_to_compute)))
for idx, (deg, val, delta, nodes, y_nodes) in enumerate(lagrange_results):
    y_vals = [lagrange_interpolation(nodes, y_nodes, xv) for xv in x_vals]
    ax1.plot(x_vals, y_vals, label=f"Degree {deg}", linewidth=2, color=colors[idx])

ax1.set_title("Lagrange Interpolation (All Degrees)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Distance (km)", fontsize=12)
ax1.set_ylabel("Temperature (°C)", fontsize=12)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

ax2.scatter(x_data, y_data, color="red", s=100, label="Sensor Data", zorder=5, 
            edgecolors='black', linewidths=2)
ax2.axvline(x_target, color="gray", linestyle="--", linewidth=2, 
            label=f"x = {x_target} km", alpha=0.7)

for idx, (deg, val, delta, nodes, table) in enumerate(newton_results):
    y_vals = [newton_interpolation(nodes, table, xv) for xv in x_vals]
    ax2.plot(x_vals, y_vals, label=f"Degree {deg}", linewidth=2, color=colors[idx])

ax2.set_title("Newton's Divided Difference (All Degrees)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Distance (km)", fontsize=12)
ax2.set_ylabel("Temperature (°C)", fontsize=12)
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interpolation_all_degrees.png', dpi=300, bbox_inches='tight')
plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

all_lag_degrees = [deg for deg, _, _, _, _ in lagrange_results]
all_lag_values = [val for _, val, _, _, _ in lagrange_results]

all_new_degrees = [deg for deg, _, _, _, _ in newton_results]
all_new_values = [val for _, val, _, _, _ in newton_results]

ax1.plot(all_lag_degrees, all_lag_values, marker='o', linewidth=2.5, markersize=10, 
         color='#2E86AB', markeredgecolor='black', markeredgewidth=1.5)
ax1.set_title("Lagrange Interpolation Values at x=45 km", fontsize=14, fontweight='bold')
ax1.set_xlabel("Polynomial Degree (k)", fontsize=12)
ax1.set_ylabel("Pₖ(45) Temperature (°C)", fontsize=12)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.set_xticks(all_lag_degrees)

for deg, val in zip(all_lag_degrees, all_lag_values):
    ax1.annotate(f'{val:.4f}°C', xy=(deg, val), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))

ax2.plot(all_new_degrees, all_new_values, marker='s', linewidth=2.5, markersize=10, 
         color='#A23B72', markeredgecolor='black', markeredgewidth=1.5)
ax2.set_title("Newton Interpolation Values at x=45 km", fontsize=14, fontweight='bold')
ax2.set_xlabel("Polynomial Degree (k)", fontsize=12)
ax2.set_ylabel("Nₖ(45) Temperature (°C)", fontsize=12)
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.set_xticks(all_new_degrees)

for deg, val in zip(all_new_degrees, all_new_values):
    ax2.annotate(f'{val:.4f}°C', xy=(deg, val), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('convergence_normal_scale.png', dpi=300, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

lag_degrees_delta = [deg for deg, _, delta, _, _ in lagrange_results if delta is not None]
lag_deltas = [delta for _, _, delta, _, _ in lagrange_results if delta is not None]

ax1.plot(lag_degrees_delta, lag_deltas, marker='o', linewidth=2.5, markersize=10, 
         color='#2E86AB', markeredgecolor='black', markeredgewidth=1.5)
ax1.set_title("Lagrange Convergence Rate (Δₖ)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Polynomial Degree (k)", fontsize=12)
ax1.set_ylabel("Δₖ = |Pₖ(45) − Pₖ₋₁(45)| (°C)", fontsize=12)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.set_xticks(lag_degrees_delta)

for deg, delta in zip(lag_degrees_delta, lag_deltas):
    ax1.annotate(f'{delta:.6f}', xy=(deg, delta), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))

new_degrees_delta = [deg for deg, _, delta, _, _ in newton_results if delta is not None]
new_deltas = [delta for _, _, delta, _, _ in newton_results if delta is not None]

ax2.plot(new_degrees_delta, new_deltas, marker='s', linewidth=2.5, markersize=10, 
         color='#A23B72', markeredgecolor='black', markeredgewidth=1.5)
ax2.set_title("Newton Convergence Rate (Δₖ)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Polynomial Degree (k)", fontsize=12)
ax2.set_ylabel("Δₖ = |Nₖ(45) − Nₖ₋₁(45)| (°C)", fontsize=12)
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.set_xticks(new_degrees_delta)

# Add value labels
for deg, delta in zip(new_degrees_delta, new_deltas):
    ax2.annotate(f'{delta:.6f}', xy=(deg, delta), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('delta_convergence_rate.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n\n" + "╔" + "═" * 78 + "╗")
print("║" + " " * 32 + "FINAL SUMMARY" + " " * 33 + "║")
print("╚" + "═" * 78 + "╝\n")

print(f"  Estimated temperature at x = {x_target} km using highest degree polynomial (Degree 8):")
print(f"  {'─' * 76}")
print(f"    Lagrange Method:     {lagrange_results[-1][1]:.10f} °C")
print(f"    Newton's Method:     {newton_results[-1][1]:.10f} °C")
print(f"    Absolute Difference: {abs(lagrange_results[-1][1] - newton_results[-1][1]):.2e} °C")
print(f"\n  Both methods produce identical results (as expected theoretically).")

print(f"\n  Convergence Analysis:")
print(f"  {'─' * 76}")
print(f"    Final convergence (Δ₈): {lagrange_results[-1][2]:.10f} °C")
print(f"    This represents the change from Degree 7 to Degree 8 polynomial.")