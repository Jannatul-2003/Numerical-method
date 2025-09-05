import matplotlib.pyplot as plt

def f(y):
    return 1-(20*20/(9.81*(((3*y)+(y**2)/2)**3)*(3+y)))

a=0.5
b=2.5
if f(a)*f(b)>0.0:
    print("The given interval is not valid")
else:
    e=1  
    i=0   
    r=(a+b)/2
    percent_relative_err_a=abs((r-a)/r)*100
    print(f"{'Iteration':>9} | {'a':>11}  | {'b':>9}    | {'r':>9}    | {'% Relative Error':>18}")
    print("-" * 65)
    errors=[]
    iterations=[]
    flag=False
    while percent_relative_err_a>=e and i<=10:
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
            print("\nApproximate root:", r)
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, errors, marker='o', linestyle='-', color='blue')
    plt.title('Convergence of Percent Relative Error')
    plt.xlabel('Iteration')
    plt.ylabel('Percent Relative Error (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()