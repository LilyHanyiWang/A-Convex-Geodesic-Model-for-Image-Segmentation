from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
import numpy as np
import time
import timesweep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import imshow
from anisdenoise import anisdenoise

def draw(gd):
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Make data.
	X = np.arange(0, 512, 1)
	Y = np.arange(0, 512, 1)
	X, Y = np.meshgrid(X, Y)
	R = gd
	Z = gd

	print(X.shape, Y.shape, Z.shape)
	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

def geodist(Im,seg,cols,rows):
	n,m = Im.shape
	mask = seg

	'''
	#make mask bigger for very slightly quicker performance
	if np.sum(seg)==1:
		indices = np.where(mask_for_1)
		indices_xp1 = indices[0]+np.ones((1,len(indices[1])))
		indices_xm1 = indices[0]-np.ones((1,len(indices[1])))
		indices_yp1 = indices[1]+np.ones((1,len(indices[1])))
		indices_ym1 = indices[1]-np.ones((1,len(indices[1])))
		mask[indices_xp1,indices[1]] = 1 
		mask[indices_xm1,indices[1]] = 1
		mask[indices[0],indices_yp1] = 1 
		mask[indices[0],indices_ym1] = 1 
		mask[indices_xp1,indices_yp1] = 1 
	'''

    #################### optional smoothing ####################
	#ims = gaussian_filter(Im,sigma=.5)
	#ims = imtgvsmooth.imtgvsmooth(Im,.05,.05,10)
	ims = anisdenoise(Im)
	
	gx,gy = np.gradient(ims)

	grad = np.multiply(gx,gx)+np.multiply(gy,gy)
	#np.savetxt('grad', grad,delimiter = ',')


	print('Compute the normal distance: ')
	eucdist = timesweep.timesweep(np.ones((n,m)),mask)

	beta=1000 
	epsi = 1e-3 
	theta=0 #in original paper, theta = 0.1

	f = epsi + beta*grad + theta*eucdist
	t = time.time()
	print('Compute the geodesic distance: ')
	gd = timesweep.timesweep(f,mask)
	print(time.time()-t)
	#np.savetxt('geodesic', gd)

	return gd