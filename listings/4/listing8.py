def hydrogen_1s(N, L):
    x = cp.linspace(-L, L, N)
    y = cp.linspace(-L, L, N)
    z = cp.linspace(-L, L, N)
    X, Y, Z_grid = cp.meshgrid(x, y, z, indexing='ij')
    r = cp.sqrt(X**2 + Y**2 + Z_grid**2)
    h1s = cp.ravel(cp.exp(-r)).reshape(N**3, 1)
    h1s /= cp.linalg.norm(h1s)
    return h1s