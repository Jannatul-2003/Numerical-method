import math
import matplotlib.pyplot as plt

def jacobi(A, b, x0, max_it, tolerance ):
    n=len(b)
    x=x0[:]
    errors=[]
    res_errors=[]
    print_header(n)

    for it in range(max_it):
        x_new=x[:]
        errors_it=[0.0]*n
        res_errors_it=0
        stop_error=0
        for i in range(n):
            if A[i][i]==0:
                raise ValueError(f"Zero diagonal element at row {i}")
            sum1=0
            for j in range(i):
                sum1+=A[i][j]*x[j]
            for j in range(i+1,n):
                sum1+=A[i][j]*x[j]
            x_new[i]=(b[i]-sum1)/A[i][i]

            errors_it[i]=abs(x_new[i]-x[i])
            stop_error=max(stop_error,errors_it[i])

            approx_b=0
            for j in range(n):
                approx_b+=A[i][j]*x_new[j]
            res_errors_it=max(res_errors_it,abs(b[i]-approx_b))
        x=x_new[:]
        res_errors.append(res_errors_it)
        errors.append(errors_it)
        print_row(it + 1, x_new, errors_it, res_errors_it)

        if stop_error<tolerance:
            break

    return x, it+1, errors, res_errors

def gauss_seidel(A, b, x0, max_it, tolerance ):
    n=len(b)
    x=x0[:]
    errors=[]
    res_errors=[]
    print_header(n)

    for it in range(max_it):
        x_new=x[:]
        errors_it=[0.0]*n
        res_errors_it=0
        stop_error=0

        for i in range(n):
            if A[i][i]==0:
                raise ValueError(f"Zero diagonal element at row {i}")
            sum1=0
            for j in range(i):
                sum1+=A[i][j]*x[j]
            for j in range(i+1,n):
                sum1+=A[i][j]*x[j]
            new_val=(b[i]-sum1)/A[i][i]
            x[i]=new_val
            errors_it[i]=abs(x_new[i]-x[i])
            stop_error=max(stop_error,errors_it[i])

            approx_b=0
            for j in range(n):
                approx_b+=A[i][j]*x_new[j]
            res_errors_it=max(res_errors_it,abs(b[i]-approx_b))
        res_errors.append(res_errors_it)
        errors.append(errors_it)
        print_row(it + 1, x_new, errors_it, res_errors_it)

        if stop_error<tolerance:
            break

    return x, it+1, errors, res_errors

def swap_rows_if_needed(A, b):
    n = len(A)
    for i in range(n):
        if A[i][i] == 0:
            for k in range(i + 1, n):
                if A[k][i] != 0:
                    A[i]=A[k]
                    A[k]=A[i]
                    b[i]=b[k]
                    b[k]=b[i]
                    break
            else:
                raise ValueError(f"Cannot fix zero diagonal element at row {i}")
    return A, b




def print_header(n):
    var_headers = [f"x{i+1}" for i in range(n)]
    err_headers = [f"err{i+1}" for i in range(n)]
    header = "Iter | " + " | ".join(f"{h:^10}" for h in var_headers) + " | " \
             + " | ".join(f"{h:^10}" for h in err_headers) + " | Residual"
    print(header)
    print("-" * len(header))


def print_row(it, x, err, res):
    row = f"{it:4d} | " + " | ".join(f"{xi:10.6f}" for xi in x) + " | " \
          + " | ".join(f"{ei:10.6f}" for ei in err) + f" | {res:10.6f}"
    print(row)

def main():
    print("=== Iterative Solvers for Linear Systems ===")
    n = int(input("Enter number of equations: "))

    print("Enter coefficient matrix A (row-wise):")
    A = [list(map(float, input().split())) for _ in range(n)]

    print("Enter constants vector b (space-separated or line by line):")
    b_input = []
    while len(b_input) < n:
        parts = input().split()
        b_input.extend(parts)
    b = [float(x) for x in b_input[:n]]

    print("Enter initial guess vector (space-separated or line by line):")
    x_input = []
    while len(x_input) < n:
        parts = input().split()
        x_input.extend(parts)
    x0 = [float(x) for x in x_input[:n]]

    max_it = int(input("Enter maximum iterations: "))
    tol = float(input("Enter tolerance: "))

    A, b = swap_rows_if_needed(A, b)

    x_jacobi, it_j, err_j, res_j = jacobi(A, b, x0, max_it, tol)
    print(f"\nJacobi converged in {it_j} iterations.")
    print("Solution:", [round(xi, 6) for xi in x_jacobi])

    x_gs, it_gs, err_gs, res_gs = gauss_seidel(A, b, x0, max_it, tol)
    print(f"\nGauss-Seidel converged in {it_gs} iterations.")
    print("Solution:", [round(xi, 6) for xi in x_gs])

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(res_j) + 1), res_j, 'o-', label="Jacobi")
    plt.plot(range(1, len(res_gs) + 1), res_gs, 's-', label="Gauss-Seidel")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Error")
    plt.title("Convergence Comparison: Jacobi vs Gauss-Seidel")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()