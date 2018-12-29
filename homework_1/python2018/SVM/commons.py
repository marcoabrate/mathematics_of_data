import numpy as np

def fmodifiedhuber(A, b, x, h):
    fx = np.zeros( A.shape[0] )
    for i in range( A.shape[0] ):
        yf = b[i] * np.dot( A[i], x )
        fx[i] = ( np.absolute( 1 -  yf ) <= h ) * ( np.square( 1 + h - yf ) / (4 * h) ) + ( yf < 1 - h ) * ( 1 - yf )
    
    fx = 0.5*fx.mean()
    return fx

def gradfmodifiedhuber(A, b, x, h):
    n = A.shape[0]
    gradfx = np.zeros(x.shape)

    for i in range(n):
        yf = b[i] * np.dot( A[i], x )
        if np.absolute( 1 - yf ) <= h:
            gradfx += ( ( yf - 1 - h ) / ( 2 * h ) ) * b[i] * np.transpose( A[i] )
        elif yf < 1 - h:
            gradfx -= b[i] * np.transpose( A[i] )

    gradfx = gradfx / (2 * n)
    return gradfx

def stogradfmodifiedhuber(A, b, x, h, i):
    yf = b[i] * np.dot( A[i], x )

    if np.absolute( 1 - yf ) <= h:
        gradfx = ( ( yf - 1 - h ) / ( 2 * h ) ) * b[i] * np.transpose( A[i] )
    elif yf < 1 - h:
        gradfx = -b[i] * np.transpose( A[i] )
    else:
        gradfx = np.zeros(x.shape)

    return gradfx

def hessfmodifiedhuber(A, b, x, h):
    n, p = A.shape
    hessf = np.zeros((p, p))

    for i in range(n):
        if np.absolute( 1 - b[i] * np.dot( A[i], x ) ) <= h:
            hessf += np.outer( A[i], A[i] )

    hessf = hessf / (4 * n * h)
    return hessf

def Oracles(b, A, lbd, h):
    """
    This function returns:
    fx: 		objective function - to compute function value at x, use fx(x)
    gradf: 		gradient mapping - to compute gradient of f at x, use gradf(x)
    hessf: 		hessian mapping - to compute hessian of f at x, use hessf(x)
    gradfsto: 	stochastic gradient mapping - to compute stochastic gradient of f_i at x, use gradfsto(x,i)
    """

    n, p = A.shape

    fx  = lambda x : 0.5*lbd*np.linalg.norm(x, 2)**2 + fmodifiedhuber(A,b,x,h)
    gradf  = lambda x: lbd*x + gradfmodifiedhuber(A,b,x,h)
    gradfsto = lambda x, i: lbd * x + stogradfmodifiedhuber(A, b, x, h, i)
    hessfx = lambda x: lbd * np.eye(p) + hessfmodifiedhuber(A, b, x, h)
    return fx, gradf, gradfsto, hessfx


def compute_error(A_test,b_test,x):
    n, err = A_test.shape[0], 0
    for i in range(n):
        if np.dot(b_test[i],np.dot(A_test[i],x)) <= 0:
           err += 1
    err = err/float(n)
    return err
