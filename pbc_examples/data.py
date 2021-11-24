import numpy as np
from utils import *
from systems_pbc import *

def load_data(beta, args):
    x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

    # remove initial and boundaty data from X_star
    t_noinitial = t[1:]
    # remove boundary at x=0
    x_noboundary = x[1:]
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)

    if 'convection' in args.system or 'diffusion' in args.system:
        u_vals = convection_diffusion(args.u0_str, args.nu, beta, args.source, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'rd' in args.system:
        u_vals = reaction_diffusion_discrete_solution(args.u0_str, args.nu, args.rho, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'reaction' in args.system:
        u_vals = reaction_solution(args.u0_str, args.rho, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    else:
        print("WARNING: System is not specified.")

    u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
    Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid

    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
    uu1 = Exact[0:1,:].T # u(x, t) at t=0
    bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
    uu2 = Exact[:,0:1] # u(-end, t)

    # generate the other BC, now at x=2pi
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t))

    u_train = uu1 # just the initial condition
    X_u_train = xx1 # (x,t) for initial condition

    return X_u_train, u_train, X_f_train, bc_lb, bc_ub, G, args.nu, beta, X_star, u_star