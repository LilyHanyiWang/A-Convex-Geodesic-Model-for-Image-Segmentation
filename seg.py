import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import copy
import math
from anisdenoise import anisdenoise
from scipy.sparse.linalg import svds

def seg(Im,lambd,theta,u0,gd):
	'''
	This function gives u in [0,1]
	Model: mu int g | grad u | + lambda int ( (z-c1)^2 - (z-c2)^2 ) v +1/2 theta int (u-v)^2 
	BRESSON CHAMBOLLE 
	'''
	print('by Roberts-Chen-Irion (2019) model -- JMIV\n')

	#Im = anisdenoise(Im)
	Im = gaussian_filter(Im,sigma=0.5)
	h,w =Im.shape

	[grad1,grad2] = np.gradient(Im)
	grad = np.multiply(grad1,grad1)+np.multiply(grad2,grad2)

	#g = 1 / (1 + 1000 * grad)
	g = np.zeros((h,w))
	for i in range(0,h):
		for j in range(0,w):
			g[i,j] = 1.0/ (1 + 100 * grad[i,j])
			g[i,j] = round(g[i,j],8)

	#Params
	rho = 1 #term on 1/{2 \theta} \int (u-v)^2
	maxit = 250
	dt = 1/8
	#stop=0.001 
	stop = 1e-4
	mu = 1 #length term (in front of regulariser)

	u=u0
	v=u0

	t=plt.title('Iterations start now...')

	Pd = theta*gd
	
	p1 = np.zeros((h,w))
	p2 = np.zeros((h,w))

	print('Iterations: START')

	for k in range(0,maxit):
		print('iter =', k)

		uold = copy.deepcopy(u)
		vold = copy.deepcopy(v)

		#np.savetxt('uold',uold)
		#np.savetxt('vold',vold)

		c1 = np.sum(np.multiply(vold, Im))   / np.sum(vold)
		c2 = np.sum(np.multiply((1-vold),Im))/ np.sum(1-vold)
		print('c1',c1)
		print('c2',c2)
		#c1 = 0.8  c2 = 0.2

		r0 = np.multiply((Im-c1),(Im-c1)) - np.multiply((Im-c2),(Im-c2))
		r = lambd*r0 + Pd
		
		#np.savetxt('r0',r0)
		#np.savetxt('r', r)

		[p1,p2,divP] = Pstuff(p1,p2,v,g,rho*mu,dt)

		resu = 0
		resv = 0

		for i in range(0,h):
			for j in range(0,w):
				# u = v - rho * mu * divP
				u[i,j] = vold[i,j] - rho * mu * divP[i,j]

				# v = max(u-rho*r,0); v = min(v,1)
				v[i,j] = u[i,j] - rho * r[i,j]
				if(v[i,j]>= 1):
					v[i,j] = 1
				elif v[i,j]<= 0:
					v[i,j] = 0
				else:
					v[i,j] = u[i,j]-rho * r[i,j]

				if (uold[i,j]!=u[i,j]):
					resu = resu + (uold[i,j] - u[i,j]) * (uold[i,j] - u[i,j])

				if (vold[i, j] != v[i, j]):
					resv = resv + (vold[i,j] - v[i,j]) * (vold[i,j] - v[i,j])

		resu = math.sqrt(resu)
		resv = math.sqrt(resv)
		
		plt.ion()
		fig, ax1 = plt.subplots(nrows=1, figsize=(6, 6))

		ax1.imshow(Im,cmap = 'gray')
		ax1.contour(u, levels=0, colors='red')
		ax1.set_title(k)

		plt.show()
		plt.pause(1)
		plt.close()
		
		Res = max(resu, resv)
		print('Res', Res)
		
		if Res < stop:
			break
		if k > 40:
			break
		### END Chambolle ###

	return u


def Pstuff(p1, p2, v, g, theta, dt):

    divP = divp(p1, p2)
    n0 = divP - v / theta
    [n1, n2] = dualgrad(n0)
    m,n = n0.shape

    D0 = np.sqrt(np.multiply(n1,n1)+np.multiply(n2,n2))
    D = 1+(dt/g)*D0

    a1 = p1 + dt * n1
    a2 = p2 + dt * n2

    newp1 = a1/D
    newp2 = a2/D
    
    return newp1, newp2, divP

def dualgrad(a):
	h,w = a.shape
	gradp1 = np.zeros((h,w))
	gradp2 = np.zeros((h,w))
	
	gradp1[0:h-2,:] = a[1:h-1,:] - a[0:h-2,:]
	gradp2[:,0:w-2] = a[:,1:w-1] - a[:,0:w-2]
	
	return gradp1,gradp2

def divp(p1,p2):
	h,w = p1.shape
	divP = np.zeros((h,w))
	
	divP[0:h-2,:] = divP[0:h-2,:] + p1[0:h-2,:]
	divP[1:h-1,:] = divP[1:h-1,:] - p1[0:h-2,:]
	divP[:,0:w-2] = divP[:,0:w-2] + p2[:,0:w-2]
	divP[:,1:w-1] = divP[:,1:w-1] - p2[:,0:w-2]
	
	return divP