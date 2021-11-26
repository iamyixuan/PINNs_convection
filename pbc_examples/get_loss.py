import torch
import numpy as np

# load model
model_path = None

# load train data

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class GetLoss:
    def __init__(self, X_u_train, X_f_train, u_train, nu, beta, rho, L, bc_lb, bc_ub, G, net,) -> None: 
        self.x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.net = net
        self.system = "convection"
        self.loss_style = "mean"
       
        self.dnn = torch.load(net).dnn

        self.u = torch.tensor(u_train, requires_grad=True).float().to(device)
        self.nu = nu
        self.beta = beta
        self.beta_final = beta
        self.rho = rho

        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

        self.L = L

    def net_u(self, x, t):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]

        if 'convection' in self.system or 'diffusion' in self.system:
            f = u_t - self.nu*u_xx + self.beta*u_x - self.G
        elif 'rd' in self.system:
            f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            f = u_t - self.rho*u + self.rho*u**2
        return f

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x

    def loss_pinn(self, verbose=True):
        """ Loss function. """
        u_pred = self.net_u(self.x_u, self.t_u)
        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub)
        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
        f_pred = self.net_f(self.x_f, self.t_f)

        if self.loss_style == 'mean':
            loss_u = torch.mean((self.u - u_pred) ** 2)
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.mean(f_pred ** 2)
        elif self.loss_style == 'sum':
            loss_u = torch.sum((self.u - u_pred) ** 2)
            loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.sum(f_pred ** 2)

        loss = loss_u + loss_b + self.L*loss_f
    
        return loss.detach().numpy(), loss_u.detach().numpy(), loss_b.detach().numpy(), loss_f.detach().numpy()

if __name__=="__main__":
    import pickle

    train_data = pickle.load(open("../history/train_data.pkl", "rb"))
    ls = GetLoss(X_u_train=train_data["X_u_train"],
                X_f_train=train_data["X_f_train"],
                u_train=train_data["u_train"],
                nu=train_data["nu"],
                beta=train_data["beta"],
                rho=train_data["rho"],
                L=train_data["L"],
                bc_lb=train_data["bc_lb"],
                bc_ub=train_data["bc_ub"],
                G=train_data["G"],
                net="./saved_models/pretrained_convection_u0sin(x)_nu0.0_beta30.0_rho0.0_Nf100_50,50,50,50,1_L2.0_source0_seed0.pt")
    loss, loss_u, loss_b, loss_f = ls.loss_pinn()

