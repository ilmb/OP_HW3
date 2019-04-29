if __name__ == '__main__':
    import numpy as np
    from cvxopt import matrix, solvers
    from scipy.linalg import norm

    Area = np.array([[1., 0.], [0., 1.], [-1., 1.], [-1., 0.], [0., -1.], [1., -1.]])
    F = np.array([1., 2.])
    y = 10.0
    P = np.transpose(Area)
    FP = np.dot(F, P)

    A = np.array([[1.], [0.]])
    B = np.array([[-1., -1., -1., -1., -1., -1.], [FP[0], FP[1], FP[2], FP[3], FP[4], FP[5]]])
    c = np.array([0., y])

    def f_x(x):
        return x[0]


    def g_z(z):
        return 0.


    def argmin_x(z, y, rho):
        #if ((sum(z) - (y[0] + 1.) / rho) > 0.0):
        ans = sum(z) - (y[0] + 1.) / rho
        return np.array([ans])


    def argmin_z(A, B, c, x, y, rho):
        P = rho * np.dot(np.transpose(B), B)
        q = (y.T.dot(B) + rho*x[0]*A.T.dot(B)-rho*c.T.dot(B)).T
        G = np.diag([-1., -1., -1., -1., -1., -1.])
        h = np.zeros(6)

        #print(q)

        G = matrix(G)
        h = matrix(h)
        P = matrix(P)
        q = matrix(q)

        sol = solvers.qp(P, q, G, h)
        #print(sol['x'])
        return np.array([sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]])


    def admm(A, B, c, rho=2, max_iter=10000):
        is_converged = False
        x = np.random.randn(1, )
        z = np.random.randn(6, )
        y = np.zeros(len(c))
        #     print x, z, y
        for iteration in range(max_iter):
            # update x
            x = argmin_x(z, y, rho)
            # update z
            z = argmin_z(A, B, c, x, y, rho)
            # update y
            y = y + rho * (np.dot(A, x) + np.dot(B, z) - c)
            # check convergence
            if norm(np.dot(A, x) + np.dot(B, z) - c) / max(norm(x), norm(z)) < 1e-12:
                is_converged = True
                break
        if not is_converged:
            print('Warning! Convergence criterion is not satisfied!')
        return np.array([x[0], z[0], z[1], z[2], z[3], z[4], z[5]])

    addm_sol = admm(A, B, c)


    print("\nAnswer:")
    t = addm_sol[0]
    x = np.array([0., 0.])
    for i in range(6):
        print("u[", i+1, "] = ", addm_sol[i+1])
        x+=addm_sol[i+1]*Area[i]
    print("Point x*: (", x[0], ", ", x[1], ")")
    print("t = ", t)