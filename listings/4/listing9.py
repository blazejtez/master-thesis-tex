def gradient_descent(goal_gradient, x0, lambd0, lr_x=1e-6, lr_lambda=1e-3, 
                     tol=1e-6, max_iter=10000):
    x = x0
    lambd = lambd0
    for i in range(max_iter):
        grad_x = goal_gradient.gradient_x(x, lambd)
        grad_lambda = goal_gradient.gradient_lambda(x)

        x_new = x - lr_x * grad_x
        if grad_lambda is not None:
            lambd_new = lambd - lr_lambda * grad_lambda
        else:
            lambd_new = lambd

        x_new = x_new / cp.linalg.norm(x_new)
        
        if grad_lambda is not None:
            if (cp.linalg.norm(x_new - x) < tol and
                    cp.linalg.norm(lambd_new - lambd) < tol):
                break
        else:
            if cp.linalg.norm(x_new - x) < tol:
                break

        x = x_new
        lambd = lambd_new

    return x, lambd