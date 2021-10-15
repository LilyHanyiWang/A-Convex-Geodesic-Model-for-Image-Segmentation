import numpy as np
import math

def timesweep(f,mask):
	n,m =f.shape
	#maxit = 1200
	inf = 1e6
	T = 1000
	u = inf*(1-mask)

	unew = np.zeros((n,m))

	res0 = []
	#stop = 5
	stop = .00005
	R = 10*stop

	count = 0
	xx = range(1,n+1)
	yy = range(1,m+1)

	orderx = np.vstack((xx,np.flip(xx),np.flip(xx),xx))
	ordery = np.vstack((yy,yy,np.flip(yy),np.flip(yy)))

	h=0.005

	while (R>stop):

		total =0
		count = count+1
		order = np.mod(count-1, 4)
		x = orderx[order, :]
		y = ordery[order, :]
		oldu = u

		for i in x:
			for j in y:

				if i == 1:
					a = u[1, j-1]
				elif i == m:
					a = u[m-2,j-1]
				else:
					temp1 = u[i-2,j-1]
					temp2 = u[i,j-1]
					a = min(temp1,temp2)

				if j==1:
					b = u[i-1,1]
				elif j ==n:
					b = u[i-1,n-2]
				else:
					b= min(u[i-1,j-2], u[i-1,j])

				if min(a,b)<T:
					cond = abs(a-b)
					if(cond >= f[i-1,j-1]*h):
						ubar = min(a,b) +f[i-1,j-1]*h
					else:
						fh = f[i-1,j-1]*h
						ubar = 1/2 * (a+b+math.sqrt((2*(fh*fh))-(a-b)*(a-b)))

					temp1= u[i-1,j-1]
					temp2 = ubar
					if (min(temp1,temp2)==temp2):
						total = total + (u[i - 1, j - 1] - ubar)**2
						total = total
						u[i-1,j-1] = ubar
		R = math.sqrt(total)

	D = ( u-np.min(u)) /( np.max(u)-np.min(u))
	print('iter =', count)

	return D