#%%

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import util

# device

print(f'Is CUDA available?: {torch.cuda.is_available()}')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# hyperparameters

torch.manual_seed(123)
trun_N = 2
layer_sizes_ubar = [2,32,32,32,1]
layer_sizes_U = [2,32,32,32,trun_N]
layer_sizes_A = [1,16,16,16,trun_N]
layer_sizes_Y = [2,64,64,64,64,trun_N]
activation = torch.tanh
step_size = 0.001
epochs = 300000
sigma_xi = 0.8
method = 'DO'

xl, xu = -np.pi, np.pi
tl, tu = 0.0, np.pi
nx, nt, nxi = 50, 50, 50
nvalx, nvalt, nvalxi = 50, 50, 50

# data

def u_(xtxi):
    return -torch.sin(xtxi[:,0:1] - sigma_xi*xtxi[:,2:3]*xtxi[:,1:2])
def ubar_t0(x):
    return -torch.sin(x)
def u_t0(x):
    u1t0 = -1/np.sqrt(np.pi)*torch.cos(x)
    u2t0 = -1/np.sqrt(np.pi)*torch.sin(x)
    return torch.cat((u1t0,u2t0),1)
def Y_t0(xi):
    Y1t0 = -xi
    Y2t0 = -np.sqrt(2)/2*(xi**2 - 1)
    return torch.cat((Y1t0,Y2t0),1)
def a_t0(t):
    a1t0 = torch.zeros_like(t)
    a2t0 = torch.zeros_like(t)
    return torch.cat((a1t0,a2t0),1)

data = {}
data['x'] = torch.linspace(xl,xu,nx)
data['xb'] = torch.Tensor([xl,xu])
data['t'] = (tu-tl)*torch.rand(nt)+tl
data['t0'] = torch.Tensor([tl])
tmp, tmp2 = np.polynomial.legendre.leggauss(nxi)
tmp = (tmp+1)/2
data['xi'] = torch.Tensor(norm.ppf(tmp))
data['wxi'] = torch.Tensor(tmp2/2)
data['wx'] = (xu-xl)/nx*torch.ones(nx)
data['wt'] = (tu-tl)/nt*torch.ones(nt)
data['wt0'] = torch.Tensor([1.])

data_val = {}
data_val['x'] = torch.linspace(xl,xu,nvalx)
data_val['xb'] = torch.Tensor([xl,xu])
data_val['t'] = torch.linspace(tl,tu,nvalt)
data_val['t0'] = torch.Tensor([tl])
tmp, tmp2 = np.polynomial.legendre.leggauss(nvalxi)
tmp = (tmp+1)/2
data_val['xi'] = torch.Tensor(norm.ppf(tmp))
data_val['wxi'] = torch.Tensor(tmp2/2)
data_val['wx'] = (xu-xl)/nx*torch.ones(nvalx)
data_val['wt'] = (tu-tl)/nt*torch.ones(nvalt)
data_val['wt0'] = torch.Tensor([1.])

xtxi = torch.cartesian_prod(data_val['x'],data_val['t'],data_val['xi'])
exact_u = u_(xtxi)

for d in data:
    data[d] = data[d].to(device)
for d in data_val:
    data_val[d] = data_val[d].to(device)

# model

class Net4(nn.Module):

    def __init__(self, layer_sizes, activation=torch.tanh):
        super(Net4, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.activation = activation

        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4])

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        x = self.l4(x)
        return x

class Net5(nn.Module):

    def __init__(self, layer_sizes, activation=torch.tanh):
        super(Net5, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.activation = activation

        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        self.l5 = nn.Linear(layer_sizes[4], layer_sizes[5])

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        x = self.l4(x)
        x = self.activation(x)
        x = self.l5(x)
        return x

net_ubar = Net4(layer_sizes_ubar, activation).to(device)
net_U = Net4(layer_sizes_U, activation).to(device)
net_A = Net4(layer_sizes_A, activation).to(device)
net_Y = Net5(layer_sizes_Y, activation).to(device)
nets = [net_ubar, net_U, net_A, net_Y]

def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):

    xtxi = torch.cartesian_prod(data['x'],data['t'],data['xi'])
    wxtxi = torch.cartesian_prod(data['wx'],data['wt'],data['wxi'])
    xt = torch.cartesian_prod(data['x'],data['t'])
    wxt = torch.cartesian_prod(data['wx'],data['wt'])
    txi = torch.cartesian_prod(data['t'],data['xi'])
    wtxi = torch.cartesian_prod(data['wt'],data['wxi'])
    t = data['t'].view(-1,1)
    wt = data['wt'].view(-1,1)
    x = data['x'].view(-1,1)
    wx = data['wx'].view(-1,1)
    xi = data['xi'].view(-1,1)
    wxi = data['wxi'].view(-1,1)
    t0 = data['t0'].view(-1,1)
    wt0 = data['wt0'].view(-1,1)
    xt0 = torch.cartesian_prod(data['x'],data['t0'])
    wxt0 = torch.cartesian_prod(data['wx'],data['wt0'])
    t0xi = torch.cartesian_prod(data['t0'],data['xi'])
    wt0xi = torch.cartesian_prod(data['wt0'],data['wxi'])
    xbt = torch.cartesian_prod(data['xb'],data['t'])

    Nx = x.shape[0]
    Nt = t.shape[0]
    Nxi = xi.shape[0]

    # MSE_w

    xtxi = xtxi.detach().requires_grad_()
    ubar = fmodel[0](xtxi[:,(0,1)], params=params_unflattened[0])
    U = fmodel[1](xtxi[:,(0,1)], params=params_unflattened[1])
    A = fmodel[2](xtxi[:,1:2], params=params_unflattened[2])
    Y = fmodel[3](xtxi[:,(1,2)], params=params_unflattened[3])

    u = ubar + (A*U*Y).sum(1).unsqueeze(1)
    Du = gradients(u,xtxi)[0]
    u_x, u_t = Du[:,0:1], Du[:,1:2]
    Nxu = - sigma_xi*xtxi[:,2:3]*u_x
    tmp = (u_t - Nxu)*wxtxi[:,2:3]
    tmp = tmp.view(Nx,Nt,Nxi).sum(2)**2
    tmp = tmp.view(Nx*Nt,1)*wxt.prod(1).unsqueeze(1)
    MSE_w = tmp.sum()

    tmp = (u_t - Nxu)*U*wxtxi[:,0:1]
    tmp = tmp.view(Nx,Nt,Nxi,trun_N).sum(0)**2
    tmp = tmp.view(Nt*Nxi,trun_N)*wtxi.prod(1).unsqueeze(1)
    MSE_w += tmp.sum()

    tmp = (u_t - Nxu)*Y*wxtxi[:,2:3]
    tmp = tmp.view(Nx,Nt,Nxi,trun_N).sum(2)**2
    tmp = tmp.view(Nx*Nt,trun_N)*wxt.prod(1).unsqueeze(1)
    MSE_w += tmp.sum()

    # MSE_0

    tmp = u_t - Nxu
    tmp = tmp**2 * wxtxi.prod(1).unsqueeze(1)
    MSE_0 = tmp.sum()

    output = [u,U,A,Y]

    # MSE_IC

    ubar = fmodel[0](xt0, params=params_unflattened[0])
    U = fmodel[1](xt0, params=params_unflattened[1])
    A = fmodel[2](t0, params=params_unflattened[2])
    Y = fmodel[3](t0xi, params=params_unflattened[3])

    tmp = ubar - ubar_t0(xt0[:,0:1])
    tmp = tmp**2 * wxt0.prod(1).unsqueeze(1)
    MSE_IC = tmp.sum()

    tmp = U - u_t0(xt0[:,0:1])
    tmp = tmp**2 * wxt0.prod(1).unsqueeze(1)
    MSE_IC += tmp.sum()/trun_N

    tmp = A - a_t0(t0)
    tmp = tmp**2 * wt0.prod(1).unsqueeze(1)
    MSE_IC += tmp.sum()/trun_N

    tmp = Y - Y_t0(t0xi[:,1:2])
    tmp = tmp**2 * wt0xi.prod(1).unsqueeze(1)
    MSE_IC += tmp.sum()/trun_N


    # MSE_BC

    ubar = fmodel[0](xbt, params=params_unflattened[0])
    U = fmodel[1](xbt, params=params_unflattened[1])
    A = fmodel[2](t, params=params_unflattened[2])
    Y = fmodel[3](txi, params=params_unflattened[3])

    tmp = ubar.view(-1,Nt)
    tmp = tmp[0:1,:] - tmp[1:2,:]
    tmp = tmp**2 * wt.view(1,Nt)
    MSE_BC = tmp.sum()

    tmp = Y.view(Nt,Nxi,trun_N) * torch.sqrt(wxi).view(1,Nxi,1)
    CYY = torch.matmul(torch.transpose(tmp,1,2),tmp) # shape: (Nt,trun_N,trun_N)
    tmp = U.view(-1,Nt,trun_N)
    tmp = tmp[0:1,:,:] - tmp[1:2,:,:]
    tmp = tmp.view(Nt,trun_N) * A
    tmp = torch.matmul(CYY,tmp.view(Nt,trun_N,1))
    tmp = tmp.view(Nt,trun_N)**2 * wt
    MSE_BC += tmp.sum()/trun_N


    # MSE_DO

    if method == 'DO':

        xt = xt.detach().requires_grad_()
        txi = txi.detach().requires_grad_()
        U = fmodel[1](xt, params=params_unflattened[1])
        Y = fmodel[3](txi, params=params_unflattened[3])

        tmp = Y*wtxi[:,1:]
        tmp = tmp.view(Nt,Nxi,trun_N).sum(1)**2
        tmp = tmp.view(Nt,trun_N) * wt
        MSE_DO = tmp.sum()/trun_N

        U_t = torch.zeros(Nx*Nt,trun_N).to(device)
        for i in range(trun_N):
            U_t[:,i:i+1] = gradients(U[:,i:i+1],xt)[0][:,1:2]
        U = U.view(Nx,Nt,trun_N)
        U_t = U_t.view(Nx,Nt,trun_N)
        U = torch.transpose(U,0,1) * wx.view(1,Nx,1) # shape: (Nt,Nx,trun_N)
        U_t = torch.transpose(U_t,0,1) # shape: (Nt,Nx,trun_N)
        tmp = torch.matmul(torch.transpose(U_t,1,2),U) # shape: (Nt,trun_N,trun_N)
        tmp = tmp**2 * wt.view(Nt,1,1)
        MSE_DO += tmp.sum()/trun_N**2

        Y_t = torch.zeros(Nt*Nxi,trun_N).to(device)
        for i in range(trun_N):
            Y_t[:,i:i+1] = gradients(Y[:,i:i+1],txi)[0][:,0:1]
        Y = Y.view(Nt,Nxi,trun_N)
        Y_t = Y_t.view(Nt,Nxi,trun_N)
        tmp = Y*Y_t*wxi.view(1,Nxi,1)
        tmp = tmp.sum(1)**2 * wt.view(Nt,1)
        MSE_DO += tmp.sum()/trun_N


    elif method == 'BO':

        xt = xt.detach().requires_grad_()
        txi = txi.detach().requires_grad_()
        U = fmodel[1](xt, params=params_unflattened[1])
        Y = fmodel[3](txi, params=params_unflattened[3])

        tmp = Y*wtxi[:,1:]
        tmp = tmp.view(Nt,Nxi,trun_N).sum(1)**2
        tmp = tmp.view(Nt,trun_N) * wt
        MSE_BO = tmp.sum()/trun_N

        U_t = torch.zeros(Nx*Nt,trun_N).to(device)
        for i in range(trun_N):
            U_t[:,i:i+1] = gradients(U[:,i:i+1],xt)[0][:,1:2]
        U = U.view(Nx,Nt,trun_N)
        U_t = U_t.view(Nx,Nt,trun_N)
        U = torch.transpose(U,0,1) * wx.view(1,Nx,1) # shape: (Nt,Nx,trun_N)
        U_t = torch.transpose(U_t,0,1) # shape: (Nt,Nx,trun_N)
        tmp = torch.matmul(torch.transpose(U_t,1,2),U) # shape: (Nt,trun_N,trun_N)
        tmp = tmp + torch.transpose(tmp,1,2)
        tmp = tmp**2 * wt.view(Nt,1,1)
        MSE_BO += tmp.sum()/trun_N**2

        Y_t = torch.zeros(Nt*Nxi,trun_N).to(device)
        for i in range(trun_N):
            Y_t[:,i:i+1] = gradients(Y[:,i:i+1],txi)[0][:,0:1]
        Y = Y.view(Nt,Nxi,trun_N) * wxi.view(1,Nxi,1)
        Y_t = Y_t.view(Nt,Nxi,trun_N)
        tmp = torch.matmul(torch.transpose(Y,1,2),Y_t) # shape: (Nt,trun_N,trun_N)
        tmp = tmp + torch.transpose(tmp,1,2)
        tmp = tmp**2 * wt.view(Nt,1,1)
        MSE_BO += tmp.sum()/trun_N


    if method == 'DO':
        ll = MSE_w + 100 * (MSE_IC + MSE_BC + MSE_DO) + 0.1*MSE_0
    elif method == 'BO':
        ll = MSE_w + 100 * (MSE_IC + MSE_BC + MSE_BO) + 0.1*MSE_0
    ll = -ll

    if torch.cuda.is_available():
        del xtxi, wxtxi, xt, wxt, txi, wtxi, t, wt, wx, wxi, t0, xt0, wxt0, t0xi, wt0xi, xbt, ubar, U, A, Y, u, Du, u_x, u_t, Nxu, tmp, CYY, U_t, Y_t, MSE_w, MSE_IC, MSE_BC, MSE_0
        if method == 'DO':
            del MSE_DO
        elif method == 'BO':
            del MSE_BO
        torch.cuda.empty_cache()

    return ll, output

# regression

params_init_val = None

params_hmc = util.sample_model_bpinns(nets, data, model_loss=model_loss, step_size=step_size, device=device, pde=True, pinns=True, epochs=epochs, params_init_val = params_init_val)

pred_list, log_prob_list = util.predict_model_bpinns(nets, params_hmc, data_val, model_loss=model_loss, pde = True)

print('\nExpected validation log probability: {:.3f}'.format(torch.stack(log_prob_list).mean()))

pred_list_u = pred_list[0].cpu().numpy()
pred_list_U = pred_list[1].cpu().numpy()
pred_list_A = pred_list[2].cpu().numpy()
pred_list_Y = pred_list[3].cpu().numpy()

# plot

wxi = data_val['wxi'].cpu().numpy()
x = data_val['x'].cpu().numpy()
t = data_val['t'].cpu().numpy()
xi = data_val['xi'].cpu().numpy()

pred_mean_u = pred_list_u.mean(0).reshape(nvalx,nvalt,nvalxi)
pred_mean_u = pred_mean_u * wxi.reshape(1,1,nvalxi)
pred_mean_u = pred_mean_u.sum(2)

exact_mean_u = exact_u.numpy().reshape(nvalx,nvalt,nvalxi)
exact_mean_u = exact_mean_u * wxi.reshape(1,1,nvalxi)
exact_mean_u = exact_mean_u.sum(2)

pred_std_u = pred_list_u.mean(0).reshape(nvalx,nvalt,nvalxi)
pred_std_u = pred_std_u**2 * wxi.reshape(1,1,nvalxi)
pred_std_u = pred_std_u.sum(2) - pred_mean_u**2

exact_std_u = exact_u.numpy().reshape(nvalx,nvalt,nvalxi)
exact_std_u = exact_std_u**2 * wxi.reshape(1,1,nvalxi)
exact_std_u = exact_std_u.sum(2) - exact_mean_u**2

pred_U = pred_list_U.mean(0).reshape(nvalx,nvalt,nvalxi,trun_N)
pred_U = pred_U[:,:,0,:]

pred_A = pred_list_A.mean(0).reshape(nvalx,nvalt,nvalxi,trun_N)
pred_A = pred_A[0,:,0,:]

pred_Y = pred_list_Y.mean(0).reshape(nvalx,nvalt,nvalxi,trun_N)
pred_Y = pred_Y[0,:,:,:]

# graph in paper

plt.figure()
plt.plot(t,pred_A[:,0],marker='s',markersize=5, label='a_1 NN-'+method)
plt.plot(t,pred_A[:,1],marker='s',markersize=5, label='a_2 NN-'+method)
plt.legend(fontsize=10)

plt.figure()
plt.plot(x,pred_U[:,nvalt-1,0],marker='s',markersize=5, label='T=pi, u_1 NN-'+method)
plt.plot(x,pred_U[:,nvalt-1,1],marker='s',markersize=5, label='T=pi, u_2 NN-'+method)
plt.legend(fontsize=10)

plt.figure()
plt.plot(xi,pred_Y[0,:,0],marker='s',markersize=5, label='T=0, Y_1 NN-'+method)
plt.plot(xi,pred_Y[0,:,1],marker='s',markersize=5, label='T=0, Y_2 NN-'+method)
plt.legend(fontsize=10)

plt.figure()
plt.plot(xi,pred_Y[int(nvalt/3)-1,:,0],marker='s',markersize=5, label='T=pi/3, Y_1 NN-'+method)
plt.plot(xi,pred_Y[int(nvalt/3)-1,:,1],marker='s',markersize=5, label='T=pi/3, Y_2 NN-'+method)
plt.legend(fontsize=10)

plt.figure()
plt.plot(xi,pred_Y[int(2*nvalt/3)-1,:,0],marker='s',markersize=5, label='T=2pi/3, Y_1 NN-'+method)
plt.plot(xi,pred_Y[int(2*nvalt/3)-1,:,1],marker='s',markersize=5, label='T=2pi/3, Y_2 NN-'+method)
plt.legend(fontsize=10)

plt.figure()
plt.plot(xi,pred_Y[nvalt-1,:,0],marker='s',markersize=5, label='T=pi, Y_1 NN-'+method)
plt.plot(xi,pred_Y[nvalt-1,:,1],marker='s',markersize=5, label='T=pi, Y_2 NN-'+method)
plt.legend(fontsize=10)

plt.figure()
plt.plot(x,pred_mean_u[:,int(2*nvalt/3)-1],marker='s',markersize=5, label='T=2pi/3, NN-'+method)
plt.plot(x,exact_mean_u[:,int(2*nvalt/3)-1],'r--', label='T=2pi/3, Exact')
plt.plot(x,pred_mean_u[:,nvalt-1],marker='^',markersize=5, label='T=pi, NN-'+method)
plt.plot(x,exact_mean_u[:,nvalt-1],'r-.', label='T=pi, Exact')
plt.legend(fontsize=10)

plt.figure()
plt.plot(x,pred_std_u[:,int(2*nvalt/3)-1],marker='s',markersize=5, label='T=2pi/3, NN-'+method)
plt.plot(x,exact_std_u[:,int(2*nvalt/3)-1],'r--', label='T=2pi/3, Exact')
plt.plot(x,pred_std_u[:,nvalt-1],marker='^',markersize=5, label='T=pi, NN-'+method)
plt.plot(x,exact_std_u[:,nvalt-1],'r-.', label='T=pi, Exact')
plt.legend(fontsize=10)
# %%
