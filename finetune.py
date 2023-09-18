
import numpy as np
import math as ma
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.kernel_ridge import KernelRidge
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model._ridge import _solve_cholesky_kernel
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

import joblib 
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
with sklearn.config_context(working_memory=8192):
    pass
import torch
import torch.nn as nn
from torch.autograd import Variable as var
import torch.utils.data as Data
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.cuda.is_available())
m = 68844 
data = np.load('dataset.npz')
cm =data['x']
jeff = data['y']
x = cm[:m]
y = jeff.reshape(-1,1)
y= y*27.2
y = y[:m]
y = np.sqrt((y**2))
print(x.shape)
x_mean = x.mean(axis=0)
x_scale = np.std(x, axis=0)
y_mean = y.mean()
y_scale = np.std(y)
print(y_mean,y_scale)
x = (x - x_mean) / x_scale

np.savetxt('0hgpgmeanedit1',x_mean, fmt='%s',delimiter=' ' )
np.savetxt('0hgpgstdedit1',x_scale, fmt='%s',delimiter=' ' )


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
x_train,x_test,y_train,y_test,x1 = var(torch.from_numpy(train_x)),var(torch.from_numpy(test_x)),var(torch.from_numpy(train_y)),var(torch.from_numpy(test_y)),var(torch.from_numpy(x1))

trainset =  Data.TensorDataset(x_train, y_train)
testset =  Data.TensorDataset(x_test, y_test)
class MultipleLayerRegressor(nn.Module):
   def __init__(self, n_feature, n_hid1, n_hid2, n_output):
       super(MultipleLayerRegressor, self).__init__()
       self.hidden = torch.nn.Linear(n_feature, n_hid1)
       self.inner1 = torch.nn.Linear(n_hid1, n_hid1)
       self.inner2 = torch.nn.Linear(n_hid1, n_hid1)
       self.inner3 = torch.nn.Linear(n_hid1, n_hid2)
       self.inner4 = torch.nn.Linear(n_hid2, n_hid2)
       self.inner5 = torch.nn.Linear(n_hid2, n_hid2)
       self.inner6 = torch.nn.Linear(n_hid2, n_hid2)
       self.out = torch.nn.Linear(n_hid2, n_output)
   def forward(self, x):
       x = torch.relu(self.hidden(x))      # activation function for hidden layer
       x = torch.relu(self.inner1(x))
       x = torch.relu(self.inner2(x))
       x = torch.relu(self.inner3(x))
       x = torch.relu(self.inner4(x))
       x = torch.relu(self.inner5(x))
       x = torch.relu(self.inner6(x))
       x = self.out(x)
       return x

class MultipleLayerRegressor1(nn.Module):
   def __init__(self, n_feature, n_hid1, n_hid2, n_output):
       super(MultipleLayerRegressor, self).__init__()
       self.hidden = torch.nn.Linear(n_feature, n_hid1)
       self.inner1 = torch.nn.Linear(n_hid1, n_hid1)
       self.inner2 = torch.nn.Linear(n_hid1, n_hid1)
       self.inner3 = torch.nn.Linear(n_hid1, n_hid2)
       self.inner4 = torch.nn.Linear(n_hid2, n_hid2)
       self.inner5 = torch.nn.Linear(n_hid2, n_hid2)
       self.inner6 = torch.nn.Linear(n_hid2, n_hid2)
       self.out = torch.nn.Linear(n_hid2, n_output) 
   def forward(self, x):
       x = torch.tanh(self.hidden(x))      # activation function for hidden layer
       x = torch.tanh(self.inner1(x))
       x = torch.tanh(self.inner2(x))
       x = torch.tanh(self.inner3(x))
       x = torch.tanh(self.inner4(x))
       x = torch.tanh(self.inner5(x))
       x = torch.tanh(self.inner6(x))
       x = self.out(x)
       return x
mlp=torch.load('mlp_.pth')
learning_rate = 2.7e-7
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=2.25E-4) 
class HomemakeLoss(torch.nn.Module):
    def __init__(self,alpha):
        super(HomemakeLoss,self).__init__()
        self.alpha=alpha
        return
    def forward(self,output,target):
        loss = torch.nn.L1Loss(reduction='mean')
        home_loss = self.alpha*loss(output,target) +(1-self.alpha)*loss(-torch.log(output),-torch.log(target))
        return torch.mean(home_loss)

loss_fn = HomemakeLoss(alpha=0.75)

bat_siz = 600

train_loader = Data.DataLoader(
    dataset=trainset,      # torch TensorDataset format
    batch_size=bat_siz,      # mini batch size
    shuffle=True,               # random shuffle for training
    )
test_loader = Data.DataLoader(
    dataset=testset,      # torch TensorDataset format
    batch_size=bat_siz,      # mini batch size
    shuffle=True,               # random shuffle for training
    )
def train_loop(dataloader, model, loss_fn, opt):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
#        print(X.shape)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
epoch = 501

for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, mlp, loss_fn, optimizer)
    test_loop(test_loader, mlp, loss_fn)
#    print(mlp(x_test).detach().numpy())
    if t % 10 == 0:
        print("Error on training set: %g ev" % (np.abs(mlp(x_train).detach().numpy() - train_y).mean()))
        print("Error on test set: %g ev" % (np.abs(mlp(x_test).detach().numpy() - test_y).mean() ))
        print("R-square on training set: %g" % r2_score(train_y, mlp(x_train).detach().numpy()))
        print("R-square on test set: %g" % r2_score(test_y, mlp(x_test).detach().numpy()))
    if t % 50 == 0:
        torch.save(mlp, 'mlplossyGPB0&90'+str(t)+'.pth')
