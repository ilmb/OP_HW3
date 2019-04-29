def makeGraph(tr):
	size_tr = tr.shape[0]
	matrix = np.zeros((N,N))
	for d in range(0,size_tr):
		matrix[tr[d,0],tr[d,1]] = 1
		matrix[tr[d,1],tr[d,2]] = 1
		matrix[tr[d,2],tr[d,0]] 	=1
		matrix[tr[d,1],tr[d,0]] = 1
		matrix[tr[d,2],tr[d,1]] = 1
		matrix[tr[d,0],tr[d,2]] = 1
return matrix
A = makeGraph(triangles_1)
B = makeGraph(triangles_2)


def proj_1(X):
	shape = X.shapeX = X.reshape((N,N))
	n = Ne = np.ones((n,1))
	return (1/n*e@e.T + X -1/n*e@e.T@X -1/n*X@e@e.T + 1/(n**2)*(e.T@X@e)*e@e.T).reshape(shape)
def proj_2(X):
	return X*(X>0)
def proj_dykstra(v, proj_1, proj_2):
	x = v
	p = np.zeros(v.shape)
	q = np.zeros(v.shape)
	k = 0
	diff = np.inf
	while diff > 1e-3 and k < 100:
		y = proj_1(x + p)
		p = x + p -y
		x_old  = x
		x = proj_2(y + q)
		q =  y + q -x
		k = k+1
		diff = norm(x -x_old)
	return x

def minimize_proj_gd(func, x0, grad, proj = lambda x: x, steepest=False):
	x = x0.copy()
	max_iter = 10000
	k=0
	x_old = x0*2
	alpha = 10
	while norm(x -x_old) > 1e-10 and k<max_iter:
		k = k+1
		p = -grad(x) # descent direction# exact search
		x_old = x     
		x = x + alpha*p
		if(func(x_old)-func(x) < 0):
			alpha = alpha/2
			else: 
				alpha = alpha*2
				x = proj(x)
	return x 
