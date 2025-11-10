import math
import matplotlib.pyplot as plt

def f(h):
    return h**3 - 10*h + 5*math.exp(-h/2) - 2

def df(h):
    return 3*h**2 - 10 - (5/2)*math.exp(-h/2)

def bisection(f, xl, xu, max_it=100, rel_er_thr=None):
    fl = f(xl)
    fu = f(xu)
    if fl * fu > 0:
        return ValueError("The interval is not valid")

    results = []
    it = 0
    old_xr = None
    while it < max_it:
        xr = (xl + xu)/2
        fr = f(xr)
        ea = None if old_xr is None else abs((xr - old_xr)/xr)*100

        results.append({
            'iteration': it+1,
            'xl': xl,
            'xu': xu,
            'xr': xr,
            'f(xl)' : fl,
            'f(xu)' : fu,
            'f(xr)': fr,
            'ea(%)': ea
        })

        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            print(f"Bisection approximate root found at x = {xr:.10f} (iter {it+1})")
            break

        if fl * fr < 0:
            xu = xr
            fu = fr
        elif fl * fr > 0:
            xl = xr
            fl = fr
        else:
            print(f"The real root is: {xr:.10f}")
            break
        old_xr = xr
        it += 1

    return results


def false_position(f, xl, xu, max_it=100, rel_er_thr=None):
    fl = f(xl)
    fu = f(xu)
    if fl * fu > 0:
        return ValueError("The interval is not valid")

    results = []
    it = 0
    old_xr = None

    while it < max_it:
        xr = (xl*fu - xu*fl)/(fu - fl)
        fr = f(xr)
        ea = None if old_xr is None else abs((xr - old_xr)/xr)*100

        results.append({
            'iteration': it+1,
            'xl': xl,
            'xu': xu,
            'xr': xr,
            'f(xl)' : fl,
            'f(xu)' : fu,
            'f(xr)': fr,
            'ea(%)': ea
        })

        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            print(f"False Position approximate root found at x = {xr:.10f} (iter {it+1})")
            break

        if fl * fr < 0:
            xu = xr
            fu = fr
        elif fl * fr > 0:
            xl = xr
            fl = fr
        else :
            print(f"The real root is: {xr:.10f}")
            break

        old_xr = xr
        it += 1

    return results

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
        ea = abs((x1 - x0)/x1)*100
        results.append({
            'iteration': it+1,
            'm_k': x0,
            'm_r': x1,
            'f(m_k)': fx0,
            "f'(m_k)": dfx0,
            'ea(%)': ea
        })
        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            print(f"Approximate Newton Raphson root found at x = {x1:.10f} on iteration {it+1} (relative error = {ea:.2e} <= threshold {rel_er_thr})")
            break
        if abs(f(x1)) < 1e-12:
            print(f"Approximate Newton Raphson root found at x = {x1:.10f} on iteration {iter+1} (|f(m_r)| < 1e-12)")
            break
        x0 = x1
        it += 1
    return results


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
        ea = abs((x2 - x1) / x2) * 100
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
            print(f"Approximate Secant root found at x = {x2:.10f} on iteration {iteration+1} (|f(x)| < 1e-12)")
            break
        if ea is not None and rel_er_thr is not None and ea <= rel_er_thr:
            print(f"Approximate Secant root found at x = {x2:.10f} on iteration {iteration+1} (relative error = {ea:.2e} <= threshold {rel_er_thr})")
            break
        x0 = x1
        f0 = f1
        x1 = x2
        f1 = f2
        iteration += 1

    return results

def print_table(results, method_name):
    """Generic function to print tables for any root-finding method"""
    if not results:
        print(f"No results to display for {method_name}")
        return
    
    columns = [key for key in results[0].keys() if key != 'iteration']
    
    width = 6 + len(columns) * 12 + (len(columns) - 1)
    
    print("\n" + "="*width)
    print(method_name.upper())
    print("="*width)
    
    header = f"{'Iter':<6} "
    for col in columns:
        header += f"{col:<12} "
    print(header)
    print("-"*width)
    
    for r in results:
        row = f"{r['iteration']:<6} "
        for col in columns:
            value = r[col]
            if value is None or (col == 'ea(%)' and value is None):
                row += f"{'N/A':<12} "
            elif isinstance(value, (int, float)):
                row += f"{value:<12.5f} "
            else:
                row += f"{str(value):<12} "
        print(row)
    
    print("="*width)



rel_err = 0.001 

bisection_results = bisection(f, 0.1, 0.4, rel_er_thr=rel_err)
print_table(bisection_results, "Bisection Method")
falsepos_results = false_position(f, 0.1, 0.4, rel_er_thr=rel_err)
print_table(falsepos_results, "False Position Method")
newton_results = newton_raphson(f, df, 1.5, rel_er_thr=rel_err)
print_table(newton_results, "Newton-Raphson Method")
secant_results = secant(f, 1.5, 2.0, rel_er_thr=rel_err)
print_table(secant_results, "Secant Method")

plt.figure(figsize=(10, 6))
def plot_errors(results, label):
    iterations = [r['iteration'] for r in results if r['ea(%)'] is not None]
    errors = [r['ea(%)'] for r in results if r['ea(%)'] is not None]
    plt.plot(iterations, errors, marker='o', label=label)

# plot_errors(bisection_results, 'Bisection')
# plot_errors(falsepos_results, 'False Position')
# plot_errors(newton_results, 'Newton-Raphson')
plot_errors(secant_results, 'Secant')

plt.xlabel('Iteration')
plt.ylabel('Approximate Error (%)')
plt.title('Convergence of Root-Finding Methods')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# import math
# import matplotlib.pyplot as plt

# def f(h):
#     return h**3 - 10*h + 5*math.exp(-h/2) - 2

# def df(h):
#     return 3*h**2 - 10 - (5/2)*math.exp(-h/2)

# def bisection(f, xl, xu, max_it=100, rel_er_thr=None):
#     fl = f(xl)
#     fu = f(xu)
#     if fl * fu > 0:
#         return ValueError("The interval is not valid")

#     results = []
#     it = 0
#     while it < max_it:
#         xr = round((xl + xu)/2, 5)
#         fr = round(f(xr), 5)
#         ea = None if it == 0 else round(abs((xr - xl)/xr)*100, 5)

#         results.append({
#             'iteration': it+1,
#             'xl': round(xl, 5),
#             'xu': round(xu, 5),
#             'xr': xr,
#             'f(xl)' : round(fl, 5),
#             'f(xu)' : round(fu, 5),
#             'f(xr)': fr,
#             'ea(%)': ea
#         })

#         if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
#             print(f"Bisection approximate root found at x = {xr:.5f} (iter {it+1})")
#             break

#         if fl * fr < 0:
#             xu = xr
#             fu = fr
#         elif fl * fr > 0:
#             xl = xr
#             fl = fr
#         else:
#             print(f"The real root is: {xr:.5f}")
#             break

#         it += 1

#     return results


# def false_position(f, xl, xu, max_it=100, rel_er_thr=None):
#     fl = f(xl)
#     fu = f(xu)
#     if fl * fu > 0:
#         return ValueError("The interval is not valid")

#     results = []
#     it = 0
#     old_xr = None
#     while it < max_it:
#         xr = round((xl*fu - xu*fl)/(fu - fl), 5)
#         fr = round(f(xr), 5)
#         ea = None if it == 0 else round(abs((xr - old_xr)/xr)*100, 5)

#         results.append({
#             'iteration': it+1,
#             'xl': round(xl, 5),
#             'xu': round(xu, 5),
#             'xr': xr,
#             'f(xl)' : round(fl, 5),
#             'f(xu)' : round(fu, 5),
#             'f(xr)': fr,
#             'ea(%)': ea
#         })

#         if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
#             print(f"False Position approximate root found at x = {xr:.5f} (iter {it+1})")
#             break

#         if fl * fr < 0:
#             xu = xr
#             fu = fr
#         elif fl * fr > 0:
#             xl = xr
#             fl = fr
#         else :
#             print(f"The real root is: {xr:.5f}")
#             break

#         old_xr = xr
#         it += 1

#     return results

# def newton_raphson(f, df, x0, max_it=100, rel_er_thr=None):
#     results = []
#     it = 0
#     while it < max_it:
#         fx0 = round(f(x0), 5)
#         dfx0 = round(df(x0), 5)
#         if dfx0 == 0:
#             print("Division by zero encountered")
#             break
#         x1 = round(x0 - fx0/dfx0, 5)
#         ea = None if it == 0 else round(abs((x1 - x0)/x1)*100, 5)
#         results.append({
#             'iteration': it+1,
#             'm_k': round(x0, 5),
#             'm_r': x1,
#             'f(m_k)': fx0,
#             "f'(m_k)": dfx0,
#             'ea(%)': ea
#         })
#         if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
#             print(f"Approximate Newton Raphson root found at x = {x1:.5f} on iteration {it+1} (relative error = {ea:.5f} <= threshold {rel_er_thr})")
#             break
#         if abs(f(x1)) < 1e-12:
#             print(f"Approximate Newton Raphson root found at x = {x1:.5f} on iteration {it+1} (|f(m_r)| < 1e-12)")
#             break
#         x0 = x1
#         it += 1
#     return results


# def secant(f, x0, x1, max_iteration=100, rel_er_thr=None):
#     f0=round(f(x0), 5)
#     f1=round(f(x1), 5)
#     results = []
#     iteration = 0
#     old_x2 = None
#     while iteration < max_iteration:
#         if abs(f1 - f0) < 1e-12:
#             print("Division by Zero or Subtract Cancellation Encountered.")
#             break
#         x2 = round(x1 - f1 * (x1 - x0) / (f1 - f0), 5)
#         f2 = round(f(x2), 5)
#         ea = None if iteration == 0 else round(abs((x2 - old_x2) / x2) * 100, 5)
#         results.append({
#             'iteration': iteration+1,
#             'x_i-1': round(x0, 5),
#             'x_i': round(x1, 5),
#             'x_i+1': x2,
#             'f(x_i-1)': f0,
#             'f(x_i)': f1,
#             'f(x_i+1)': f2,
#             'ea(%)': ea
#         })

#         if abs(f2) < 1e-12:
#             print(f"Approximate Secant root found at x = {x2:.5f} on iteration {iteration+1} (|f(x)| < 1e-12)")
#             break
#         if ea is not None and rel_er_thr is not None and ea <= rel_er_thr:
#             print(f"Approximate Secant root found at x = {x2:.5f} on iteration {iteration+1} (relative error = {ea:.5f} <= threshold {rel_er_thr})")
#             break
#         x0 = x1
#         f0 = f1
#         x1 = x2
#         f1 = f2
#         old_x2 = x2
#         iteration += 1

#     return results

# def print_table(results, method_name):
#     """Generic function to print tables for any root-finding method"""
#     if not results:
#         print(f"No results to display for {method_name}")
#         return
    
#     columns = [key for key in results[0].keys() if key != 'iteration']
    
#     width = 6 + len(columns) * 12 + (len(columns) - 1)
    
#     print("\n" + "="*width)
#     print(method_name.upper())
#     print("="*width)
    
#     header = f"{'Iter':<6} "
#     for col in columns:
#         header += f"{col:<12} "
#     print(header)
#     print("-"*width)
    
#     for r in results:
#         row = f"{r['iteration']:<6} "
#         for col in columns:
#             value = r[col]
#             if value is None or (col == 'ea(%)' and value is None):
#                 row += f"{'N/A':<12} "
#             elif isinstance(value, (int, float)):
#                 row += f"{value:<12.5f} "
#             else:
#                 row += f"{str(value):<12} "
#         print(row)
    
#     print("="*width)



# rel_err = 0.001 

# bisection_results = bisection(f, 0.1, 0.4, rel_er_thr=rel_err)
# print_table(bisection_results, "Bisection Method")
# falsepos_results = false_position(f, 0.1, 0.4, rel_er_thr=rel_err)
# print_table(falsepos_results, "False Position Method")
# newton_results = newton_raphson(f, df, 1.5, rel_er_thr=rel_err)
# print_table(newton_results, "Newton-Raphson Method")
# secant_results = secant(f, 1.5, 2.0, rel_er_thr=rel_err)
# print_table(secant_results, "Secant Method")

# plt.figure(figsize=(10, 6))
# def plot_errors(results, label):
#     iterations = [r['iteration'] for r in results if r['ea(%)'] is not None]
#     errors = [r['ea(%)'] for r in results if r['ea(%)'] is not None]
#     plt.plot(iterations, errors, marker='o', label=label)

# plot_errors(bisection_results, 'Bisection')
# plot_errors(falsepos_results, 'False Position')
# plot_errors(newton_results, 'Newton-Raphson')
# plot_errors(secant_results, 'Secant')

# plt.xlabel('Iteration')
# plt.ylabel('Approximate Error (%)')
# plt.title('Convergence of Root-Finding Methods')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.show()


