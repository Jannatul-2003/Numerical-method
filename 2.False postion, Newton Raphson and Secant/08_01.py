import math
import matplotlib.pyplot as plt

def f1(m):
    return 0.654*m*(1-math.exp(-150/m))-36

def false_position(f, xl, xu, max_it=100, rel_er_thr=None):
    fl = f(xl)
    fu = f(xu)
    if fl * fu > 0:
        raise ValueError("f(xl) and f(xu) must have opposite signs")
    
    results = []
    it = 0
    while it < max_it:
        xr = (xl*fu - xu*fl)/(fu - fl)
        fr = f(xr)
        ea = None if it == 0 else abs((xr - old_xr)/xr)*100
        results.append({
            'iteration': it+1,
            'xl': xl,
            'xu': xu,
            'xr': xr,
            'f(xr)': fr,
            'ea(%)': ea
        })
        if fr == 0 or (ea is not None and rel_er_thr is not None and ea < rel_er_thr):
            print(f"Approximate root found at x = {xr:.10f} on iteration {it+1} (relative error = {ea:.2e} <= threshold {rel_er_thr})")
            break
        if fl*fr < 0:
            xu = xr
            fu = fr
        else:
            xl = xr 
            fl = fr
        old_xr = xr
        it += 1
    return results

results_fp = false_position(f1, 40, 80, max_it=50, rel_er_thr=0.001)

print(f"{'Iter':>4} {'xl':>14} {'xu':>14} {'xr':>14} {'f(xr)':>16} {'εa(%)':>14}")
for r in results_fp:
    ea_str = f"{r['ea(%)']:14.10f}" if r['ea(%)'] is not None else " " * 14
    print(f"{r['iteration']:>4} {r['xl']:14.10f} {r['xu']:14.10f} {r['xr']:14.10f} {r['f(xr)']:16.10f}{ea_str}")

iters = [r['iteration'] for r in results_fp if r['ea(%)'] is not None]
errors = [r['ea(%)'] for r in results_fp if r['ea(%)'] is not None]

plt.plot(iters, errors, 'ro-', label='False Position')
plt.xlabel('Iteration')
plt.ylabel('Approximate Error (%)')
plt.title('False Position Convergence')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
