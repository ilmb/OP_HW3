if __name__ == '__main__':
    from cvxopt import matrix, solvers
    import numpy as np
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt


    #####   G   ######
    G = np.zeros((4, 2))
    G[0,0] = -1
    G[1,0] = -1
    G[2,0] = -1
    G[3,0] = -1
    G[0,1] = 2
    G[1,1] = 1
    G[2,1] = -2
    G[3,1] = -1

    G = matrix(G)
    print("G=")
    print(G)
    ##################

    #####   C   ######
    c = matrix([1.,0.])
    print("c=")
    print(c)
    ##################

    #####   h   ######
    h = matrix([1.,1.,-1.,-1.])
    print("h=")
    print(h)
    ##################

    sol = solvers.lp(c, G, h)
    solution = sol['x']

    print(solution[1])


    ######## Визуализация #########
    
    fig, ax = plt.subplots()
    patches = []

    x = np.linspace(-5, 5, 100)
    y = 0.5 * x
    plt.plot(x, y, '-r', label='Фx=y')
    plt.plot(2 * solution[1], 1 * solution[1], '*')
    plt.plot(1, 1, '*')
    # patches.append(Polygon(p2, True))
    patches.append(Polygon([[8, 1], [9, 2], [9, 2]], True))

    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 10])

    plt.axis('equal')
    plt.show()
