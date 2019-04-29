
if __name__ == '__main__':
    from cvxopt import matrix, solvers
    import numpy as np
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    ##### исходные данные #####
    P = matrix([[1.,0.],[0.,1.],[-1.,1.],[-1.,0.],[0.,-1.],[1.,-1.]])
    F = matrix([[1.],[2.]])
    y = 10
    ###########################

    dim,n=P.size
    FP=F*P

    #####   C   ######
    c=np.zeros(n+1)
    c[0]=1
    c = matrix(c)
    print("c=")
    print(c)
    ##################

    #####   G   ######
    G = np.zeros((n+5,n+1))
    G[0,0]=1; G[0,1:]=-1
    G[1,:]=-G[0,:]

    for i in range(0,n+1):
        G[2+i,i]=-1

    G[n+3,1:]=FP
    G[n+4,:]=-G[n+3,:]
    G = matrix(G)
    print("G=")
    print(G)
    ##################

    #####   h   ######
    h = np.zeros(n+5)
    h[n+3]=y
    h[n+4]=-y
    h=matrix(h)
    print("h=")
    print(h)
    ##################


    sol = solvers.lp(c, G, h)
    solution=sol['x']
    t=solution[0]
    print("\n\n\n||x||=",t)

    xStar=np.zeros((dim,1))
    for i in range(n):
        xStar+=P[:,i]*solution[1+i]

    print("\nx*=",xStar)

    ######## Визуализация #########
    p2=P.trans()*t

    xOpt = [1,2]
    yOpt = [3,4]

    fig, ax = plt.subplots()
    patches = []

    x = np.linspace(-5, 5, 100)
    y = -0.5 * x + 5
    plt.plot(x, y, '-r', label='Фx=y')
    plt.plot(xStar[0],xStar[1],'*b')
    patches.append(Polygon(p2, True))
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
