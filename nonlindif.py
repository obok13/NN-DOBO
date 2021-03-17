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

torch.manual_seed(1234)
trun_N = 8
dim_xi = 19
layer_sizes_ubar = [2,32,32,32,1]
layer_sizes_U = [2,64,64,64,trun_N]
layer_sizes_A = [1,4,4,4,1]
layer_sizes_Y = [1+dim_xi,64,64,64,trun_N]
activation = torch.tanh
step_size = 0.001
epochs = 300000
a = 0.1
b = 0.5
sigma_g = 1
lc = 0.1
method = 'BO'

xl, xu = -1.0, 1.0
tl, tu = 0.0, 1.0
# nx, nt, nxi = 51, 50, 1000
# nvalx, nvalt, nvalxi = 51, 50, 1000
nx, nt, nxi = 50, 30, 200
nvalx, nvalt, nvalxi = 50, 30, 200

# data

def ubar_t0(x):
    return -torch.sin(np.pi*x)
def u_t0(x):
    return torch.sin(np.pi*(torch.arange(trun_N).to(device)+1).view(1,-1)*(x+1)/2)
def Y_t0(xi):
    return xi[:,:trun_N]
def a_t0(t):
    return torch.zeros_like(t).repeat(1,trun_N)

data = {}
data['x'] = torch.linspace(xl,xu,nx)
data['xb'] = torch.Tensor([xl,xu])
data['t'] = (tu-tl)*torch.rand(nt)+tl
data['t0'] = torch.Tensor([tl])
data['xi'] = torch.randn([nxi,dim_xi])
data['wxi'] = 1/nxi*torch.ones(nxi)
data['wx'] = (xu-xl)/nx*torch.ones(nx)
data['wt'] = (tu-tl)/nt*torch.ones(nt)
data['wt0'] = torch.Tensor([1.])

data_val = {}
data_val['x'] = torch.linspace(xl,xu,nvalx)
data_val['xb'] = torch.Tensor([xl,xu])
data_val['t'] = torch.linspace(tl,tu,nvalt)
data_val['t0'] = torch.Tensor([tl])
data_val['xi'] = torch.randn([nvalxi,dim_xi])
data_val['wxi'] = 1/nvalxi*torch.ones(nvalxi)
data_val['wx'] = (xu-xl)/nvalx*torch.ones(nvalx)
data_val['wt'] = (tu-tl)/nvalt*torch.ones(nvalt)
data_val['wt0'] = torch.Tensor([1.])

for d in data:
    data[d] = data[d].to(device)
for d in data_val:
    data_val[d] = data_val[d].to(device)

# KLD of GP

KL_z, KL_w = np.polynomial.legendre.leggauss(dim_xi+20)
covmat = sigma_g**2 * np.exp( -(KL_z.reshape(-1,1) - KL_z.reshape(1,-1))**2 / lc**2 )
covmat_data = sigma_g**2 * np.exp( -(data['x'].cpu().numpy().reshape(-1,1) - KL_z.reshape(1,-1))**2 / lc**2 )
covmat_val = sigma_g**2 * np.exp( -(data_val['x'].cpu().numpy().reshape(-1,1) - KL_z.reshape(1,-1))**2 / lc**2 )

sqrtW = np.diag(np.sqrt(KL_w))
WcovmatW = np.dot(np.dot(sqrtW,covmat),sqrtW)
lamk, uk = np.linalg.eig(WcovmatW)
lamk = np.real(lamk[:dim_xi])
uk = np.real(uk[:,:dim_xi])
fk_data = 1/np.sqrt(lamk).reshape(1,-1)*np.dot(covmat_data,uk*np.sqrt(KL_w).reshape(-1,1))
fk_val = 1/np.sqrt(lamk).reshape(1,-1)*np.dot(covmat_val,uk*np.sqrt(KL_w).reshape(-1,1))
fk_data = torch.Tensor(fk_data).to(device)
fk_val = torch.Tensor(fk_val).to(device)
f_data = (1 - data['x'].view(1,-1)**2) * (1 + torch.mm(data['xi'],fk_data.T))
f_val = (1 - data_val['x'].view(1,-1)**2) * (1 + torch.mm(data_val['xi'],fk_val.T))

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

net_ubar = Net4(layer_sizes_ubar, activation).to(device)
net_U = Net4(layer_sizes_U, activation).to(device)
nets = [net_ubar,net_U]
for _ in range(trun_N):
    nets.append(Net4(layer_sizes_A, activation).to(device))
net_Y = Net4(layer_sizes_Y, activation).to(device)
nets.append(net_Y)

def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):

    xt = torch.cartesian_prod(data['x'],data['t'])
    wxt = torch.cartesian_prod(data['wx'],data['wt'])
    t = data['t'].view(-1,1)
    wt = data['wt'].view(-1,1)
    x = data['x'].view(-1,1)
    wx = data['wx'].view(-1,1)
    xi = data['xi']
    wxi = data['wxi'].view(-1,1)
    t0 = data['t0'].view(-1,1)
    wt0 = data['wt0'].view(-1,1)
    xt0 = torch.cartesian_prod(data['x'],data['t0'])
    wxt0 = torch.cartesian_prod(data['wx'],data['wt0'])
    xbt = torch.cartesian_prod(data['xb'],data['t'])

    Nx = x.shape[0]
    Nt = t.shape[0]
    Nxi = xi.shape[0]

    xtxi = torch.cartesian_prod(data['x'],data['t'],data['xi'][:,0])[:,0:2]
    xtxi = torch.cat((xtxi,data['xi'].repeat(Nx*Nt,1)),1)
    wxtxi = torch.cartesian_prod(data['wx'],data['wt'],data['wxi'])

    txi = torch.cartesian_prod(data['t'],data['xi'][:,0])[:,0:1]
    txi = torch.cat((txi,data['xi'].repeat(Nt,1)),1)
    wtxi = torch.cartesian_prod(data['wt'],data['wxi'])

    t0xi = torch.cat((t0.repeat(Nxi,1),xi),1)
    wt0xi = torch.cartesian_prod(data['wt0'],data['wxi'])



    # MSE_w

    xtxi = xtxi.detach().requires_grad_()
    ubar = fmodel[0](xtxi[:,(0,1)], params=params_unflattened[0])
    U = fmodel[1](xtxi[:,(0,1)], params=params_unflattened[1])
    A = fmodel[2](xtxi[:,1:2], params=params_unflattened[2])
    for i in range(trun_N-1):
        A = torch.cat((A,fmodel[i+3](xtxi[:,1:2], params=params_unflattened[i+3])),1)
    Y = fmodel[trun_N+2](xtxi[:,1:], params=params_unflattened[trun_N+2])

    u = ubar + (A*U*Y).sum(1).unsqueeze(1)
    Du = gradients(u,xtxi)[0]
    u_x, u_t = Du[:,0:1], Du[:,1:2]
    u_xx = gradients(u_x,xtxi)[0][:,0:1]
    Nxu = a*u_xx + b*u**2 + ( f_data.T.view(Nx,1,Nxi) + torch.zeros(1,Nt,1).to(device) ).reshape(-1,1)

    tmp = (u_t - Nxu)*wxtxi[:,2:3].prod(1).unsqueeze(1)
    tmp = tmp.view(Nx,Nt,Nxi).sum(2)**2
    tmp = tmp.view(Nx*Nt,1)*wxt.prod(1).unsqueeze(1)
    MSE_w = tmp.sum()

    tmp = (u_t - Nxu)*U*wxtxi[:,0:1]
    tmp = tmp.view(Nx,Nt,Nxi,trun_N).sum(0)**2
    tmp = tmp.view(Nt*Nxi,trun_N)*wtxi.prod(1).unsqueeze(1)
    MSE_w += tmp.sum()

    tmp = (u_t - Nxu)*Y*wxtxi[:,2:3].prod(1).unsqueeze(1)
    tmp = tmp.view(Nx,Nt,Nxi,trun_N).sum(2)**2
    tmp = tmp.view(Nx*Nt,trun_N)*wxt.prod(1).unsqueeze(1)
    MSE_w += tmp.sum()


    # MSE_0

    tmp = u_t - Nxu
    tmp = tmp**2 * wxtxi.prod(1).unsqueeze(1)
    MSE_0 = tmp.sum()

    output = [u,U,A,Y,ubar]

    # MSE_IC

    ubar = fmodel[0](xt0, params=params_unflattened[0])
    U = fmodel[1](xt0, params=params_unflattened[1])
    A = fmodel[2](t0, params=params_unflattened[2])
    for i in range(trun_N-1):
        A = torch.cat((A,fmodel[i+3](t0, params=params_unflattened[i+3])),1)
    Y = fmodel[trun_N+2](t0xi, params=params_unflattened[trun_N+2])

    tmp = ubar - ubar_t0(xt0[:,0:1])
    tmp = tmp**2 * wxt0.prod(1).unsqueeze(1)
    MSE_IC = tmp.sum()

    tmp = U - u_t0(xt0[:,0:1])
    tmp = tmp**2 * wxt0.prod(1).unsqueeze(1)
    MSE_IC += tmp.sum()/trun_N

    tmp = A - a_t0(t0)
    tmp = tmp**2 * wt0.prod(1).unsqueeze(1)
    MSE_IC += tmp.sum()/trun_N

    tmp = Y - Y_t0(t0xi[:,1:])
    tmp = tmp**2 * wt0xi.prod(1).unsqueeze(1)
    MSE_IC += tmp.sum()/trun_N


    # MSE_BC

    ubar = fmodel[0](xbt, params=params_unflattened[0])
    U = fmodel[1](xbt, params=params_unflattened[1])
    A = fmodel[2](t, params=params_unflattened[2])
    for i in range(trun_N-1):
        A = torch.cat((A,fmodel[i+3](t, params=params_unflattened[i+3])),1)
    Y = fmodel[trun_N+2](txi, params=params_unflattened[trun_N+2])

    tmp = ubar.view(-1,Nt)
    tmp = tmp**2 * wt.view(1,Nt)
    MSE_BC = tmp.sum()

    tmp = Y.view(Nt,Nxi,trun_N) * torch.sqrt(wxi.prod(1).view(1,Nxi,1))
    CYY = torch.matmul(torch.transpose(tmp,1,2),tmp) # shape: (Nt,trun_N,trun_N)
    tmp = U.view(-1,Nt*trun_N)
    tmp = tmp.T.view(Nt,trun_N,-1)
    tmp = tmp * A.view(Nt,trun_N,1)
    tmp = torch.matmul(CYY,tmp)
    tmp = tmp.view(Nt,-1)**2 * wt
    MSE_BC += tmp.sum()/trun_N


    # MSE_DO

    if method == 'DO':

        xt = xt.detach().requires_grad_()
        txi = txi.detach().requires_grad_()
        U = fmodel[1](xt, params=params_unflattened[1])
        Y = fmodel[trun_N+2](txi, params=params_unflattened[trun_N+2])

        tmp = Y*wtxi[:,1:].prod(1).unsqueeze(1)
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
        tmp = Y*Y_t*wxi.prod(1).view(1,Nxi,1)
        tmp = tmp.sum(1)**2 * wt.view(Nt,1)
        MSE_DO += tmp.sum()/trun_N


    elif method == 'BO':

        xt = xt.detach().requires_grad_()
        txi = txi.detach().requires_grad_()
        U = fmodel[1](xt, params=params_unflattened[1])
        Y = fmodel[trun_N+2](txi, params=params_unflattened[trun_N+2])

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
        Y = Y.view(Nt,Nxi,trun_N) * wxi.prod(1).view(1,Nxi,1)
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
        del xtxi, wxtxi, xt, wxt, txi, wtxi, t, wt, wx, wxi, t0, xt0, wxt0, t0xi, wt0xi, xbt, ubar, U, A, Y, u, Du, u_x, u_xx, u_t, Nxu, tmp, CYY, U_t, Y_t, MSE_w, MSE_IC, MSE_BC, MSE_0
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
pred_list_ubar = pred_list[4].cpu().numpy()

# exact for u

xxi = torch.cartesian_prod(data_val['x'],data_val['xi'][:,0])[:,0:1]
xxi = torch.cat((xxi,data_val['xi'].repeat(nvalx,1)),1)
Nht = 100
ht = (tu-tl)/(nvalt-1)/Nht
hx = (xu-xl)/(nvalx-1)
exact_u = ubar_t0(xxi[:,0:1]).view(nvalx,1,nvalxi)
D2 = torch.zeros(nvalx,nvalx).to(device)
for i in range(nvalx):
    for j in range(nvalx):
        if i==0:
            D2[i,i] = 1.
        elif i==nvalx-1:
            D2[i,i] = 1.
        else:
            if i==j:
                D2[i,j] = -2/hx**2
            elif abs(i-j)==1:
                D2[i,j] = 1/hx**2

for i in range((nvalt-1)*Nht):
    if i<2:
        tmp = a*torch.matmul(D2,exact_u[:,i,:]) + b*exact_u[:,i,:]**2 + f_val.T
        tmp = exact_u[:,i,:] + ht*tmp
        exact_u = torch.cat((exact_u,tmp.view(nvalx,1,nvalxi)),1)
    else:
        tmp1 = a*torch.matmul(D2,exact_u[:,i,:]) + b*exact_u[:,i,:]**2 + f_val.T
        tmp2 = a*torch.matmul(D2,exact_u[:,i-1,:]) + b*exact_u[:,i-1,:]**2 + f_val.T
        tmp3 = a*torch.matmul(D2,exact_u[:,i-2,:]) + b*exact_u[:,i-2,:]**2 + f_val.T
        tmp = exact_u[:,i,:] + ht/12 * (23*tmp1 - 16*tmp2 + 5*tmp3)
        exact_u = torch.cat((exact_u,tmp.view(nvalx,1,nvalxi)),1)

exact_u = exact_u[:,::Nht,:].cpu().numpy()

# plot

wxi = data_val['wxi'].cpu().numpy()
x = data_val['x'].cpu().numpy()
t = data_val['t'].cpu().numpy()

pred_mean_u = pred_list_u.mean(0).reshape(nvalx,nvalt,nvalxi)
pred_mean_u = pred_mean_u * wxi.reshape(1,1,nvalxi)
pred_mean_u = pred_mean_u.sum(2)

pred_std_u = pred_list_u.mean(0).reshape(nvalx,nvalt,nvalxi)
pred_std_u = pred_std_u**2 * wxi.reshape(1,1,nvalxi)
pred_std_u = pred_std_u.sum(2) - pred_mean_u**2
pred_std_u = pred_std_u**0.5

exact_mean_u = exact_u * wxi.reshape(1,1,nvalxi)
exact_mean_u = exact_mean_u.sum(2)

exact_std_u = exact_u**2 * wxi.reshape(1,1,nvalxi)
exact_std_u = exact_std_u.sum(2) - exact_mean_u**2
exact_std_u = exact_std_u**0.5

pred_A = pred_list_A.mean(0).reshape(nvalx,nvalt,nvalxi,trun_N)[0,:,0,:]

pred_ubar = pred_list_ubar.mean(0).reshape(nvalx,nvalt,nvalxi)[:,:,0]

# graph in paper

att1 = int(1*(nvalt-1)/10)
att2 = int(10*(nvalt-1)/10)

plt.figure()
plt.plot(x,pred_mean_u[:,att1],marker='s',markersize=5, label='T=0.1, NN-'+method)
plt.plot(x,pred_mean_u[:,att2],marker='^',markersize=5, label='T=1.0, NN-'+method)
plt.plot(x,exact_mean_u[:,att1],'r--', label='T=0.1, Exact')
plt.plot(x,exact_mean_u[:,att2],'r-.', label='T=1.0, Exact')
plt.legend(fontsize=10)

plt.figure()
plt.plot(x,pred_std_u[:,att1],marker='s',markersize=5, label='T=0.1, NN-'+method)
plt.plot(x,pred_std_u[:,att2],marker='^',markersize=5, label='T=1.0, NN-'+method)
plt.plot(x,exact_std_u[:,att1],'r--', label='T=0.1, Exact')
plt.plot(x,exact_std_u[:,att2],'r-.', label='T=1.0, Exact')
plt.legend(fontsize=10)

plt.figure()
plt.plot(t,abs(pred_A),marker='s',markersize=5, label='NN-'+method)

plt.figure()
plt.plot(x,pred_mean_u[:,att2],'b-', label='T=1, mean, NN-'+method)
plt.fill_between(x,pred_mean_u[:,att2] - pred_std_u[:,att2], pred_mean_u[:,att2] + pred_std_u[:,att2],'b', label='T=1, std, NN-'+method, alpha=0.2)
plt.plot(x,exact_mean_u[:,att2],'r-', label='T=1, mean, reference')
plt.fill_between(x,exact_mean_u[:,att2] - exact_std_u[:,att2], exact_mean_u[:,att2] + exact_std_u[:,att2],'r', label='T=1, std, reference', alpha=0.2)
plt.legend(fontsize=10)
# %%
