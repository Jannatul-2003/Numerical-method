import math
import matplotlib.pyplot as plt

def f3(m):
    return 0.654*m*(1-math.exp(-150/m))-36

def secant(f, x0, x1, max_iteration=100, rel_er_thr=None):
    f0=f(x0)
    f1=f(x1)
    results = []
    iteration = 0
    while iteration < max_iteration:
        if abs(f1 - f0) < 1e-12:
            print("Division by Zero or Subtract Cancellation Encountered.")
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        ea = None if iteration == 0 else abs((x2 - old_x2) / x2) * 100
        results.append({
            'iteration': iteration+1,
            'x_i-1': x0,
            'x_i': x1,
            'x_i+1': x2,
            'f(x_i-1)': f0,
            'f(x_i)': f1,
            'f(x_i+1)': f2,
            'ea(%)': ea
        })

        if abs(f2) < 1e-12:
            print(f"Approximate root found at x = {x2:.10f} on iteration {iteration+1} (|f(x)| < 1e-12)")
            break
        if ea is not None and rel_er_thr is not None and ea <= rel_er_thr:
            print(f"Approximate root found at x = {x2:.10f} on iteration {iteration+1} (relative error = {ea:.2e} <= threshold {rel_er_thr})")
            break
        x0 = x1
        f0 = f1
        x1 = x2
        f1 = f2
        old_x2 = x2
        iteration += 1

    return results

results = secant(f3, 30, 40, max_iteration=100, rel_er_thr=0.001)


print(f"{'Iteration':>4} {'x_i-1':>14} {'x_i':>14} {'x_i+1':>14} {'f(x_i-1)':>14} {'f(x_i)':>14} {'f(x_i+1)':>14} {'Ea(%)':>14}")
for r in results:
    ea_str = f"{r['ea(%)']:.10f}" if r['ea(%)'] is not None else "None"
    print(f"{r['iteration']:4d} {r['x_i-1']:14.10f} {r['x_i']:14.10f} {r['x_i+1']:14.10f} "
          f"{r['f(x_i-1)']:14.10f} {r['f(x_i)']:14.10f} {r['f(x_i+1)']:14.10f} {ea_str:>14}")

iterations = [r['iteration'] for r in results if r['ea(%)'] is not None]
errors = [r['ea(%)'] for r in results if r['ea(%)'] is not None]

plt.plot(iterations, errors, 'go-', label='Secant')
plt.xlabel('Iteration')
plt.ylabel('Approximate Error (%)')
plt.title('Secant Convergence')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
