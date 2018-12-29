import numpy as np

def fmodifiedhuyer(xt, y, w, h):
    fx = np.zeros(xt.shape[0])
    for i in range(xt.shape[0]):
        yf = y[i] * np.dot(xt[i], w)
        fx[i] = (np.absolute(1 - yf) <= h) * (np.square(1 + h - yf) / (4 * h)) + (yf < 1 - h) * (1 - yf)

    fx = 0.5 * fx.mean()
    return fx


def gradfmodifiedhuber(xt, y, w, h):
    n = xt.shape[0]
    gradfx = np.zeros(w.shape)

    for i in range(n):
        yf = y[i] * np.dot(xt[i], w)
        if np.absolute(1 - yf) <= h:
            gradfx += ((yf - 1 - h) / (2 * h)) * y[i] * np.transpose(xt[i])
        elif yf < 1 - h:
            gradfx -= y[i] * np.transpose(xt[i])

    gradfx = gradfx / (2 * n)
    return gradfx

def Oracles(y, xt, lbd, h):
    """
    This function returns:
    fx: 		objective function - to compute function value at w, use fx(w)
    gradf: 		gradient mapping - to compute gradient of f at w, use gradf(w)
    hessf: 		hessian mapping - to compute hessian of f at w, use hessf(w)
    gradfsto: 	stochastic gradient mapping - to compute stochastic gradient of f_i at x, use gradfsto(x,i)
    """

    n, p = xt.shape

    fx  = lambda w : 0.5*lbd*np.linalg.norm(w, 2)**2 + fmodifiedhuber(xt,y,w,h)
    gradf  = lambda w: lbd*w + gradfmodifiedhuber(xt,y,w,h)
    gradfsto = lambda w, i: lbd * w + stogradfmodifiedhuber(xt, y, w, h, i)
    hessfx = lambda w: lbd * np.eye(p) + hessfmodifiedhuber(xt, y, w, h)
    return fx, gradf, gradfsto, hessfx