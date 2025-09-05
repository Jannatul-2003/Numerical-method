import matplotlib.pyplot as plt

def f(x):
    return 225 + 82*x - 90*x**2 + 44*x**3 - 8*x**4 + 0.7*x**5

a=-1.2
b=-1.0
if f(a)*f(b)>0:
    print("The given interval is not valid")
else:
    e=0.05  
    i=0   
    r=(a+b)/2
    percent_relative_err_a=abs((r-a)/r)*100
    print(f"{'Iteration':>9} | {'a':>12}  | {'b':>10}    | {'r':>10}    | {'% Relative Error':>18}")
    print("-" * 65)
    errors=[]
    iterations=[]
    flag=False
    while percent_relative_err_a>e:
        r=(a+b)/2
        percent_relative_err_a = abs((r-a)/r)*100
        errors.append(percent_relative_err_a)
        iterations.append(i+1)
        print(f"{i+1:9d} | {a:10.10f} | {b:10.10f} | {r:10.10f} | {percent_relative_err_a:17.10f}%")
        if f(a)*f(r)<0:
            b=r
        elif f(b)*f(r)<0:
            a=r
        else:
            print("\nThe root is:", r)
            flag=True
            break
        i+=1
    else:
        if not flag:
            print("\nApproximate root (within tolerance):", r)
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, errors, marker='o', linestyle='-', color='blue')
    plt.title('Convergence of Percent Relative Error')
    plt.xlabel('Iteration')
    plt.ylabel('Percent Relative Error (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()