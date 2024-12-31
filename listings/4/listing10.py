x = x0
    lambd = lambd0

    beta1 = 0.9
    beta2 = 0.99
    epsilon = 1e-8

    m_x = cp.zeros_like(x)
    v_x = cp.zeros_like(x)
    if lambd is not None and lambd.size > 0:
        m_lambda = cp.zeros_like(lambd)
        v_lambda = cp.zeros_like(lambd)
    else:
        m_lambda = None
        v_lambda = None

    t = 0
    for i in range(max_iter):
        t = i + 1

        grad_x = goal_gradient.gradient_x(x, lambd)
        grad_lambda = goal_gradient.gradient_lambda(x)
        prev_eigenvalue = goal_gradient.objective_function(x, lambd)

        m_x = beta1 * m_x + (1 - beta1) * grad_x
        v_x = beta2 * v_x + (1 - beta2) * cp.square(grad_x)
        m_hat_x = m_x / (1 - beta1 ** t)
        v_hat_x = v_x / (1 - beta2 ** t)
        x_new = x - lr_x * m_hat_x / (cp.sqrt(v_hat_x) + epsilon)

        if grad_lambda is not None:
            m_lambda = beta1 * m_lambda + (1 - beta1) * grad_lambda
            v_lambda = beta2 * v_lambda + (1 - beta2) * cp.square(grad_lambda)
            m_hat_lambda = m_lambda / (1 - beta1 ** t)
            v_hat_lambda = v_lambda / (1 - beta2 ** t)
            lambd_new = lambd + lr_lambda * m_hat_lambda / (cp.sqrt(v_hat_lambda) + epsilon)
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