import numpy as np
from numpy import linalg as LA
from SVM.commons import Oracles, compute_error
from SVM.algorithms import  GD, GDstr, AGD, AGDstr, LSGD, LSAGD, AGDR, LSAGDR, QNM, NM, SGD, SAG, SVR
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def main():
    training, testing = np.load('dataset/training.npz'), np.load('dataset/testing.npz')
    A, b = training['A'], training['b']
    A_test, b_test = testing['A'], testing['b']

    print(68 * '*')
    print('Linear Support Vector Machine:')
    print('Modified Huber Loss + Ridge Regularizer')
    print('Dataset Size : {} x {}'.format(A.shape[0], A.shape[1]))
    print(68 * '*')

    # Choose the solvers you want to call
    GD_option = 1
    GDstr_option = 1
    AGD_option = 1
    AGDstr_option = 1
    LSGD_option = 1
    LSAGD_option = 1
    AGDR_option = 1
    LSAGDR_option = 1
    QNM_option = 1
    NM_option = 1
    SGD_option = 1
    SAG_option = 1
    SVR_option = 1


    # Set parameters and solve numerically with GD, AGD, AGDR, LSGD, LSAGD, LSAGDR.
    print('Numerical solution process is started:')
    n, p = A.shape
    sigma = 1e-4
    h = 0.5
    # fs_star = 0.045102342 # old
    fs_star = 0.037003410 # new

    parameter = {}
    parameter['Lips'] = LA.norm(np.transpose(A), 2)*LA.norm(A, 2)/(4 * n * h) + sigma
    parameter['strcnvx'] = sigma
    parameter['x0'] = np.zeros(p)
    parameter['Lmax'] = 0
    for i in range(n):
        parameter['Lmax']= np.maximum(parameter['Lmax'], (1 / (4*h)) * LA.norm(A[i], 2)*LA.norm(A[i], 2))
    parameter['Lmax'] += sigma

    fx, gradf, gradfsto, hessf = Oracles(b, A, sigma, h)
    x, info, error = {}, {}, {}

    # first-order methods
    parameter['maxit'] = 8000
    if GD_option:
        x['GD'], info['GD'] = GD(fx, gradf, parameter)
        error['GD'] = compute_error(A_test, b_test, x['GD'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['GD']

    parameter['maxit'] = 8000
    if GDstr_option:
        x['GDstr'], info['GDstr'] = GDstr(fx, gradf, parameter)
        error['GDstr'] = compute_error(A_test, b_test, x['GDstr'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['GDstr']

    parameter['maxit'] = 4000
    if AGD_option:
        x['AGD'], info['AGD'] = AGD(fx, gradf, parameter)
        error['AGD'] = compute_error(A_test, b_test, x['AGD'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['AGD']

    parameter['maxit'] = 2000
    if AGDstr_option:
        x['AGDstr'], info['AGDstr'] = AGDstr(fx, gradf, parameter)
        error['AGDstr'] = compute_error(A_test, b_test, x['AGDstr'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['AGDstr']

    parameter['maxit'] = 500
    if AGDR_option:
        x['AGDR'], info['AGDR'] = AGDR(fx, gradf, parameter)
        error['AGDR'] = compute_error(A_test, b_test, x['AGDR'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['AGDR']

    parameter['maxit'] = 450
    if LSGD_option:
        x['LSGD'], info['LSGD'] = LSGD(fx, gradf, parameter)
        error['LSGD'] = compute_error(A_test, b_test, x['LSGD'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['LSGD']

    parameter['maxit'] = 400
    if LSAGD_option:
        x['LSAGD'], info['LSAGD'] = LSAGD(fx, gradf, parameter)
        error['LSAGD'] = compute_error(A_test, b_test, x['LSAGD'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['LSAGD']

    parameter['maxit'] = 100
    if LSAGDR_option:
        x['LSAGDR'], info['LSAGDR'] = LSAGDR(fx, gradf, parameter)
        error['LSAGDR'] = compute_error(A_test, b_test, x['LSAGDR'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['LSAGDR']


    # second-order methods
    parameter['maxit'] = 20
    if NM_option:
        x['NM'], info['NM'] = NM(fx, gradf, hessf, parameter)
        error['NM'] = compute_error(A_test, b_test, x['NM'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['NM']

    parameter['maxit'] = 200
    if QNM_option:
        x['QNM'], info['QNM'] = QNM(fx, gradf, parameter)
        error['QNM'] = compute_error(A_test, b_test, x['QNM'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['QNM']


    # stochastic methods
    parameter['no0functions'] = n
    parameter['maxit'] = 5*n
    if SGD_option:
        x['SGD'], info['SGD'] = SGD(fx, gradfsto, parameter)
        error['SGD'] = compute_error(A_test, b_test, x['SGD'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['SGD']

    if SAG_option:
        x['SAG'], info['SAG'] = SAG(fx, gradfsto, parameter)
        error['SAG'] = compute_error(A_test, b_test, x['SAG'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['SAG']

    parameter['maxit'] = int(0.3*n)
    if SVR_option:
        x['SVR'], info['SVR'] = SVR(fx, gradf, gradfsto, parameter)
        error['SVR'] = compute_error(A_test, b_test, x['SVR'])
        print('Error w.r.t 0-1 loss: %0.9f') % error['SVR']


    print('Numerical solution process is completed.')
    colors_ = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = [colors_[name] for name in colors_.keys()]
    
    # plot figures
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)

    for key in x.keys():
        if key not in ['SGD', 'SAG', 'SVR']:
            ax1.plot(np.array(range(1, info[key]['iter']+1)), info[key]['fx'] - fs_star, color=colors[x.keys().index(key)], lw=2, label=key)
    ax1.legend()
    ax1.set_ylim(1e-9, 1e0)
    ax1.set_xlabel('#iterations')
    ax1.set_ylabel(r'$f(\mathbf{x}^k) - f^\star$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid()

    ax2 = plt.subplot(1, 2, 2)

    for key in x.keys():
        if key in ['GD', 'SGD', 'SAG', 'SVR']:
            if key =='GD':
                ax2.plot(np.array(range(info[key]['iter'])), info[key]['fx'] - fs_star, lw=2, label=key, marker='o')
            else:
                ax2.plot(np.array(range(info[key]['iter']))/float(n),info[key]['fx'] - fs_star, lw=2, label=key)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(1e-4, 1e0)
    ax2.legend()
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel(r'$f(\mathbf{x}^k) - f^\star$')
    # ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


