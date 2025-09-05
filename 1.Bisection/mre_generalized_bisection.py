import math

def bisection_method(f, a, b, true_value=None, significant_digits=4):
    if f(a) * f(b) > 0:
        print("The given interval is not valid. f(a) and f(b) must have opposite signs.")
        return None

    e = 10 ** (-significant_digits)
    N = math.ceil((math.log(b - a) - math.log(e)) / math.log(2))
    print(f"\nNumber of iterations required to achieve {significant_digits} significant digits: {N}\n")
    
    print(f"{'Iter':^5} | {'a':^10} | {'b':^10} | {'r':^12} | {'%Err(a)':^12} | {'%Err(t)':^12}")
    print("-" * 70)

    for i in range(1, N+1):
        r = (a + b) / 2
        percent_relative_err_a = abs((r - a) / r) * 100
        percent_relative_err_t = abs((true_value - r) / true_value) * 100 if true_value is not None else None

        print(f"{i:^5} | {a:^10.6f} | {b:^10.6f} | {r:^12.6f} | {percent_relative_err_a:^12.6f} | {percent_relative_err_t if percent_relative_err_t is not None else 'N/A':^12}")

        if f(a) * f(r) < 0:
            b = r
        elif f(b) * f(r) < 0:
            a = r
        else:
            print(f"\nThe root is found exactly: r = {r}")
            return r

    print(f"\nApproximate root after {N} iterations: r = {r}")
    return r

def f(x):
    return x**3 - x - 2

a = float(input("Enter the value of a: "))
b = float(input("Enter the value of b: "))
t = float(input("Enter the true value (optional, you can put 0 if unknown): "))
n = int(input("Enter the number of significant digits: "))

bisection_method(f, a, b, true_value=t, significant_digits=n)
