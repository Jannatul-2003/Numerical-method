import math
import matplotlib.pyplot as plt


def f2(m):
    return 0.654*m*(1-math.exp(-150/m))-36

def df2(m):
    return 0.654 * (1 - math.exp(-150/m) - (150/m) * math.exp(-150/m))

def newton_raphson(f, df, x0, max_it=100, rel_er_thr=None):
    results = []
    it = 0
    while it < max_it:
        fx0 = f(x0)
        dfx0 = df(x0)
        if dfx0 == 0:
            print("Division by zero encountered")
            break
        x1 = x0 - fx0/dfx0
        ea = None if it == 0 else abs((x1 - x0)/x1)*100
        results.append({
            'iteration': it+1,
            'm_k': x0,
            'm_r': x1,
            'f(m_k)': fx0,
            "f'(m_k)": dfx0,
            'ea(%)': ea
        })
        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            print(f"Approximate root found at x = {x1:.10f} on iteration {it+1} (relative error = {ea:.2e} <= threshold {rel_er_thr})")
            break
        if abs(f2(x1)) < 1e-12:
            print(f"Approximate root found at x = {x1:.10f} on iteration {iter+1} (|f(m_r)| < 1e-12)")
            break
        x0 = x1
        it += 1
    return results

results = newton_raphson(f2, df2, 40, max_it=50, rel_er_thr=0.001)

print(f"{'Iter':>4} {'m_k':>14} {'m_r':>14} {'f(m_k)':>14} {'f\'(m_k)':>14} {'Ea(%)':>14}")
for r in results:
    ea_str = f"{r['ea(%)']:.10f}" if r['ea(%)'] is not None else "None"
    print(f"{r['iteration']:4d} {r['m_k']:14.10f} {r['m_r']:14.10f} {r['f(m_k)']:14.10f} {r['f\'(m_k)']:14.10f} {ea_str:>14}")

iterations = [r['iteration'] for r in results if r['ea(%)'] is not None]
errors = [r['ea(%)'] for r in results if r['ea(%)'] is not None]

plt.plot(iterations, errors, 'bo-', label='Newton-Raphson')
plt.xlabel('Iteration')
plt.ylabel('Approximate Error (%)')
plt.title('Newton-Raphson Convergence')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
