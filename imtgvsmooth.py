import numpy as np
import math

def imtgvsmooth( Y, alpha, beta, nite ):
	sy,sx,sc = Y.shape

	global rho,eta
	rho, eta = 1,1

	################filter kernels#################

	#differential kernels
	global FKh,FKv,FKh_,FKv_
	Kh = np.matrix([1, -1, 0]) #forward difference by convolution
	Kv = np.matrix([1, -1, 0]).T

	#FFTed filters
	FKh = psf2otf( Kh, [sy, sx] )
	FKv = psf2otf( Kv, [sy, sx] )

	FKh_ = conj(FKh) #for computation in the Fourier domain
	FKv_ = conj(FKv)
	FKl = FKh_*FKh + FKv_*FKv

	global FY
	FY = fft2( Y )

	#########pixel-wise inverse by using the image-wise adjugate mathod
	global iA11,iA12,iA13,iA21,iA22,A23, iA31, iA32, iA33
	A11 = 1+rho*FKl
	A12 = -rho*FKh_
	A13 = -rho*FKv_
	A21 = -rho*FKh
	A22 = rho+eta*FKl
	A23 = eta*FKv*FKh_
	A31 = -rho*FKv
	A32 = eta*FKh*FKv_
	A33 = rho+eta*FKl

	detA = 1/ ((A11*A22*A33) + (A12*A32*A13) + (A31*A12*A23)- (A11*A32*A23) - (A31*A22*A13) - (A21*A12*A33))

	iA11 = (A22*A33 - A23*A32) * detA
	iA12 = (A13*A32 - A12*A33) * detA
	iA13 = (A12*A23 - A13*A22) * detA
	iA21 = (A23*A31 - A21*A33) * detA
	iA22 = (A11*A33 - A13*A31) * detA
	iA23 = (A13*A21 - A11*A23) * detA
	iA31 = (A21*A32 - A22*A31) * detA
	iA32 = (A12*A31 - A11*A32) * detA
	iA33 = (A11*A22 - A12*A21) * detA

	####################Initialization###############
	X = Y

	Z1h = diff_circ( Y, 2, 'forward' )
	Z1v = diff_circ( Y, 1, 'forward' )
	Z2h = np.zeros(sy,sx)
	Z2d = np.zeros(sy,sx)
	Z2v = np.zeros(sy,sx)

	U1h = np.zeros(sy,sx)
	U1v = np.zeros(sy,sx)
	U2h = np.zeros(sy,sx)
	U2d = np.zeros(sy,sx)
	U2v = np.zeros(sy,sx)

	##################### ADMM ##################
	# For simplicity, the stopping criteria of the ADMM is omitted.

	for t in range(1,nite+1):
		# solve J
		[X, Qh, Qv] = opt_X( Z1h, Z1v, U1h, U1v, Z2h, Z2d, Z2v, U2h, U2d, U2v )

		# solve Z1
		Xu = diff_circ( X, 2, 'forward' )
		Xv = diff_circ( X, 1, 'forward' )
		
		T1h = (Xu - Qh)
		T1v = (Xv - Qv)
		
		Z1h = T1h + U1h
		Z1v = T1v + U1v

		[Z1h, Z1v] = shrinkage( alpha/rho , Z1h, Z1v )

		# solve Z2
		T2h = diff_circ( Qh, 2, 'backward' )
		T2d = diff_circ( Qh, 1, 'backward' ) + diff_circ( Qv, 2, 'backward' )
		T2v = diff_circ( Qv, 1, 'backward' )
		
		Z2h = T2h + U2h
		Z2d = T2d + U2d
		Z2v = T2v + U2v

		[Z2h, Z2d, Z2v] = shrinkage( beta/eta, Z2h, Z2d, Z2v )

		# update Uz
		U1h = U1h + (T1h - Z1h)
		U1v = U1v + (T1v - Z1v)
		
		# update U2
		U2h = U2h + (T2h - Z2h)
		U2d = U2d + (T2d - Z2d)
		U2v = U2v + (T2v - Z2v)

def diff_circ( I, dim, ori ):

	if dim == 1: #virtical
	    return {
	        'forward': diff( np.matrix([I,I[1,:,:]]).T, [], 1 ),
	        'backward': -diff( np.matrix([I[end,:,:],I]).T, [], 1 ),
	    }.get(x, diff( np.matrix([I,I[1,:,:]]).T, [], 1 ))

	else: #dim == 2 #horizontal
		return {
	        'forward': diff( [I, I[:,1,:]], [], 2 ),
	        'backward': -diff( [I[:,end,:], I], [], 2 ),
	    }.get(x, diff( [I, I[:,1,:]], [], 2 ))

def shrinkage( t, varargin):
	#[X1,...,XN] = shrinkage( t, X1,...,XN );

	varargout = cell( nargout, 1 )
	n = len(varargin)

	# pixel-wise norm
	T = varargin[0]
	N = np.multiply(T,T)
	for i in range(1,n):
		T = varargin[i]
		N = N + np.multiply(T,T)

	N = math.sqrt(N)
	S = np.max(1 - t/N, 0)

	for i in range(0,n):
		varargout[i] = np.multiply(S,varargin[i])

	return varargout