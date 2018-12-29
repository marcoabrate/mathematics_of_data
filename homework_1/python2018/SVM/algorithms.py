import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve


def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                 - gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Gradient Descent')

    maxit = parameter['maxit']

    # Initialize x and alpha.

    x = parameter['x0']
    L = parameter['Lips']
    alpha = 1 / L

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    info['iter'] = maxit
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter):
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """

    print(68 * '*')
    print('Gradient Descent  with strong convexity')

    maxit = parameter['maxit']

    # Initialize x and alpha.

    x = parameter['x0']
    L = parameter['Lips']
    mu = parameter['strcnvx']
    alpha = 2 / (L + mu)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient')

    maxit = parameter['maxit']

    # Initialize x, y, t and alpha.

    x = parameter['x0']
    y = x
    t = 1
    L = parameter['Lips']
    alpha = 1 / L

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_next = y - alpha * gradf(y)
        t_next = (1 + np.sqrt(4 * (t ** 2) + 1)) / 2
        y = x_next + ((t - 1) / t_next) * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        t = t_next

    return x, info


# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with strong convexity')

    maxit = parameter['maxit']

    # Initialize x, y, alpha and gamma.

    x = parameter['x0']
    y = x
    L = parameter['Lips']
    mu = parameter['strcnvx']
    alpha = 2 / (L + mu)
    gamma = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_next = y - alpha * gradf(y)
        y = x_next + gamma * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    return x, info


# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Gradient Descent with line search')

    maxit = parameter['maxit']

    # Initialize x and alpha.

    x = parameter['x0']
    L = parameter['Lips']
    alpha = 1 / L

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        # LINE SEARCH
        L_next0 = 0.5 * L
        d = -gradf(x)
        i = 0
        while (fx(x + (d / ((2 ** i) * L_next0))) >
               (fx(x) - (np.linalg.norm(d) / np.sqrt((2 ** (i + 1)) * L_next0)) ** 2)):
            i = i + 1

        L_next = (2 ** i) * L_next0
        alpha = 1 / L_next

        x_next = x + alpha * d

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        L = L_next

    return x, info


# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
        """
    print(68 * '*')
    print('Accelerated Gradient with line search')

    maxit = parameter['maxit']

    # Initialize x, y, t and alpha.

    x = parameter['x0']
    y = x
    t = 1
    L = parameter['Lips']
    alpha = 1 / L

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        # LINE SEARCH
        L_next0 = 0.5 * L
        d = -gradf(y)
        i = 0
        while (fx(y + (d / ((2 ** i) * L_next0))) >
               (fx(y) - (np.linalg.norm(d) / np.sqrt((2 ** (i + 1)) * L_next0)) ** 2)):
            i = i + 1

        L_next = (2 ** i) * L_next0
        alpha = 1 / L_next

        # UPDATING THE NEXT ITERATIONS
        x_next = y + alpha * d
        t_next = (1 + np.sqrt(4 * (L_next / L) * (t ** 2) + 1)) / 2
        y = x_next + ((t - 1) / t_next) * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        L = L_next

    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with restart')

    maxit = parameter['maxit']

    # Initialize x, y, t, alpha and find the initial function value (fval).

    x = parameter['x0']
    y = x
    t = 1
    L = parameter['Lips']
    alpha = 1 / L

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_next = y - alpha * gradf(y)

        # RESTART
        if (fx(x) < fx(x_next)):
            t = 1
            y = x
            x_next = y - alpha * gradf(y)
        t_next = (1 + np.sqrt(4 * (t ** 2) + 1)) / 2
        y = x_next + ((t - 1) / t_next) * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with line search + restart')

    maxit = parameter['maxit']

    # Initialize x, y, t and alpha.

    x = parameter['x0']
    y = x
    t = 1
    L = parameter['Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        # LINE SEARCH
        L_next0 = 0.5 * L
        d = -gradf(y)
        i = 0
        while (fx(y + (d / ((2 ** i) * L_next0))) >
               (fx(y) - (np.linalg.norm(d) / np.sqrt((2 ** (i + 1)) * L_next0)) ** 2)):
            i = i + 1

        L_next = (2 ** i) * L_next0
        alpha = 1 / L_next

        # UPDATING THE NEXT ITERATIONS
        x_next = y + alpha * d

        # RESTART
        if (fx(x) < fx(x_next)):
            t = 1
            y = x
            x_next = y - alpha * gradf(y)
        t_next = (1 + np.sqrt(4 * (L_next / L) * (t ** 2) + 1)) / 2
        y = x_next + ((t - 1) / t_next) * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        L = L_next

    return x, info


def QNM(fx, gradf, parameter):
    """
    Function:  [x, info] = QNM (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the quasi-Newton method with BFGS update.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Quasi Newton Method')
    maxit = parameter['maxit']

    # Initialize x and alpha.

    x = parameter['x0']
    B = np.identity(len(x))
    alpha = 10.0

    theta = 64
    key = 0.1

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        gradient = gradf(x)

        d = -B.dot(gradient)
        alpha_next0 = theta * alpha
        i = 0
        while (fx(x + (alpha_next0 * d) / (2 ** i)) >
               (fx(x) + key * (alpha_next0 / (2 ** (i + 1))) * gradient.T.dot(d))):
            i = i + 1

        alpha_next = alpha_next0 / (2 ** i)

        x_next = x + alpha_next * d

        s = x_next - x
        v = gradf(x_next) - gradient

        # UPDATING B
        Bv = B.dot(v)
        B_next = B - np.outer(Bv, Bv) / (v.T.dot(Bv)) + np.outer(s, s) / (s.T.dot(v))

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        alpha = alpha_next
        B = B_next

    return x, info


# Newton
def NM(fx, gradf, hessf, parameter):
    """
    Function:  [x, info] = NM (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the Newton method.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping

    :return: x, info
    """

    print(68 * '*')

    print('Newton Method')
    maxit = parameter['maxit']

    # Initialize x and alpha.

    x = parameter['x0']
    alpha = 10.0

    theta = 64
    key = 0.1

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        hessian = hessf(x)
        gradient = gradf(x)
        d = spla.spsolve(hessian, -gradient)
        alpha_next0 = theta * alpha
        i = 0
        while (fx(x + (alpha_next0 * d) / (2 ** i)) >
               (fx(x) + key * (alpha_next0 / (2 ** (i + 1))) * gradient.T.dot(d))):
            i = i + 1

        alpha_next = alpha_next0 / (2 ** i)

        x_next = x + alpha_next * d

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        alpha = alpha_next

    return x, info


def SGD(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SGD(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lmax']       - Maximum of Lipschitz constant of all f_i.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               parameter['no0functions']    - Number of functions.
               fx                      - objective function
               gradfsto               - stochastic gradient mapping

    :return: x, info
    """
    print(68 * '*')
    print('Stochastic Gradient Descent')
    maxit = parameter['maxit']

    # Initialize x and alpha and other.

    x = parameter['x0']
    n = parameter['no0functions']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        i_k = randint(n)
        
        # TODO: does this if statement make sens !?!?!?!?!?
        if (iter <= n):
            alpha_k = 1 / float(iter + 1)
        x_next = x - alpha_k * gradfsto(x, i_k)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SAG(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SAG(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic averaging gradient algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lmax']       - Maximum of Lipschitz constant of all f_i.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               parameter['no0functions']    - Number of functions.
               fx                      - objective function
               gradfsto               - stochastic gradient mapping
    """
    print(68 * '*')
    print('Stochastic Averaging Gradient')
    maxit = parameter['maxit']

    # Initialize.

    x = parameter['x0']
    n = parameter['no0functions']
    v = np.zeros((len(x), n))
    Lmax = parameter['Lmax']
    alpha = 1 / (16 * Lmax)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        i_k = randint(n)
        v[:, i_k] = gradfsto(x, i_k)
        x_next = x - (alpha / n) * np.sum(v, axis=1)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic gradient with variance reduction algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lmax']       - Maximum of Lipschitz constant of all f_i.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               parameter['no0functions']    - Number of functions.
               fx                      - objective function
               gradf                  - gradient mapping
               gradfsto               - stochastic gradient mapping
    """
    print(68 * '*')
    print('Stochastic Gradient Descent with variance reduction')
    maxit = parameter['maxit']

    # Initialize.

    x = parameter['x0']
    n = parameter['no0functions']
    Lmax = parameter['Lmax']
    gamma = 0.01 / Lmax
    q = int(1000 * Lmax)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_tilde_l = x
        sum_x_tilde_l = 0
        v_k = gradf(x)
        for l in range(q):
            i_l = randint(n)
            v_l = gradfsto(x_tilde_l, i_l) - gradfsto(x, i_l) + v_k
            x_tilde_l = x_tilde_l - gamma * v_l
            sum_x_tilde_l = sum_x_tilde_l + x_tilde_l

        x_next = (1/float(q)) * sum_x_tilde_l

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info