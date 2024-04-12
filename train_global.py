import warnings
import os
from model_function import *
from model_utils import *
from utils import *
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from torchdiffeq import odeint as odeint
import matplotlib
matplotlib.use('Agg')
import argparse
import sys
import time
import torch
torch.manual_seed(42)
torch.cuda.empty_cache() 
import torch.optim as optim
import random
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import sys

set_seed(42)
cwd = os.getcwd()
#data_path = {'z500':str(cwd) + '/era5_data/geopotential_500/*.nc','t850':str(cwd) + '/era5_data/temperature_850/*.nc'}
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"adaptive_heun","euler"]
parser = argparse.ArgumentParser('ClimODE')

parser.add_argument('--solver', type=str, default="euler", choices=SOLVERS)
parser.add_argument('--atol', type=float, default=5e-3)
parser.add_argument('--rtol', type=float, default=5e-3)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--scale', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--spectral', type=int, default=0,choices=[0,1])
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=1e-5)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_time_scale= slice('2006','2016')
val_time_scale = slice('2016','2016')
test_time_scale = slice('2017','2018')

paths_to_data = [str(cwd) + '/era5_data/geopotential_500/*.nc',str(cwd) + '/era5_data/temperature_850/*.nc',str(cwd) + '/era5_data/2m_temperature/*.nc',str(cwd) + '/era5_data/10m_u_component_of_wind/*.nc',str(cwd) + '/era5_data/10m_v_component_of_wind/*.nc']
const_info_path = [str(cwd) +  '/era5_data/constants/constants_5.625deg.nc']
levels = ["z","t","t2m","u10","v10"]
paths_to_data = paths_to_data[0:5]
levels = levels[0:5]
assert len(paths_to_data) == len(levels), "Paths to different type of data must be same as number of types of observations"
print("############################ Data is loading ###########################")
Final_train_data = 0
Final_val_data = 0
Final_test_data = 0
max_lev = []
min_lev = []

for idx,data in enumerate(paths_to_data):
    Train_data,Val_data,Test_data,time_steps,lat,lon,mean,std,time_stamp = get_train_test_data_without_scales_batched(data,train_time_scale,val_time_scale,test_time_scale,levels[idx],args.spectral)  
    max_lev.append(mean)
    min_lev.append(std)
    if idx==0: 
        Final_train_data = Train_data
        Final_val_data = Val_data
        Final_test_data = Test_data
    else:
        Final_train_data = torch.cat([Final_train_data,Train_data],dim=2)
        Final_val_data = torch.cat([Final_val_data,Val_data],dim=2)
        Final_test_data = torch.cat([Final_test_data,Test_data],dim=2)


print("Length of training data",len(Final_train_data))
print("Length of validation data",len(Final_val_data))
print("Length of testing data",len(Final_test_data))
const_channels_info,lat_map,lon_map = add_constant_info(const_info_path)
H,W = Train_data.shape[3],Train_data.shape[4]
Train_loader = DataLoader(Final_train_data[2:],batch_size=args.batch_size,shuffle=False,pin_memory=False)
Val_loader = DataLoader(Final_val_data[2:],batch_size=args.batch_size,shuffle=False,pin_memory=False)
Test_loader = DataLoader(Final_test_data[2:],batch_size=args.batch_size,shuffle=False,pin_memory=False)
time_loader = DataLoader(time_steps[2:],batch_size=args.batch_size,shuffle=False,pin_memory=False)
time_idx_steps = torch.tensor([i for i in range(365*4)]).view(-1,1)
time_idx = DataLoader(time_idx_steps[2:],batch_size=args.batch_size,shuffle=False,pin_memory=False)
#Model declaration
num_years = len(range(2006,2016))
model = Climate_encoder_free_uncertain(len(paths_to_data),2,out_types=len(paths_to_data),method=args.solver,use_att=True,use_err=True,use_pos=False).to(device)
#model.apply(weights_init_uniform_rule)
param = count_parameters(model)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

best_loss = float('inf')
train_best_loss = float('inf')
best_epoch = float('inf')
print("############################ Data is loaded, Fitting the velocity #########################")

get_gauss_kernel((32,64),lat,lon)
kernel = torch.from_numpy(np.load(str(cwd) +"/kernel.npy"))
#breakpoint()
fit_velocity(time_idx,time_loader,Final_train_data,Train_loader,torch.device('cpu'),num_years,paths_to_data,args.scale,H,W,types='train_10year_2day_mm',vel_model=Optim_velocity,kernel=kernel,lat=lat,lon=lon)
fit_velocity(time_idx,time_loader,Final_val_data,Val_loader,torch.device('cpu'),1,paths_to_data,args.scale,H,W,types='val_10year_2day_mm',vel_model=Optim_velocity,kernel=kernel,lat=lat,lon=lon)
fit_velocity(time_idx,time_loader,Final_test_data,Test_loader,torch.device('cpu'),2,paths_to_data,args.scale,H,W,types='test_10year_2day_mm',vel_model=Optim_velocity,kernel=kernel,lat=lat,lon=lon)

vel_train,vel_val = load_velocity(['train_10year_2day_mm','val_10year_2day_mm'])
print("############################ Velocity loaded, Model starts to train #########################")
print(model)
print("####################### Total Parameters",param ,"################################")

for epoch in range(args.niters):
    total_train_loss = 0
    val_loss = 0
    test_loss = 0
    #RMSD = []
    #breakpoint()
    if epoch == 0:
        var_coeff = 0.001
    else:
        var_coeff = 2*scheduler.get_last_lr()[0]
    
    for entry,(time_steps,batch) in enumerate(zip(time_loader,Train_loader)):
        optimizer.zero_grad()
        data = batch[0].to(device).view(num_years,1,len(paths_to_data)*(args.scale+1),H,W)
        past_sample = vel_train[entry].view(num_years,2*len(paths_to_data)*(args.scale+1),H,W).to(device)
        model.update_param([past_sample,const_channels_info.to(device),lat_map.to(device),lon_map.to(device)])
        t = time_steps.float().to(device).flatten()
        mean,std = model(t,data)
        loss = nll(mean,std,batch.float().to(device),lat,var_coeff)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()    
        print("Loss for batch is ",loss.item())
        if torch.isnan(loss) : 
            print("Quitting due to Nan loss")
            quit()
        total_train_loss = total_train_loss + loss.item()

    lr_val = scheduler.get_last_lr()[0]
    scheduler.step()
    print("|Iter ",epoch," | Total Train Loss ", total_train_loss,"|")
  
    for entry,(time_steps,batch) in enumerate(zip(time_loader,Val_loader)):
        data = batch[0].to(device).view(1,1,len(paths_to_data)*(args.scale+1),H,W)
        past_sample = vel_val[entry].view(1,2*len(paths_to_data)*(args.scale+1),H,W).to(device)
        model.update_param([past_sample,const_channels_info.to(device),lat_map.to(device),lon_map.to(device)])
        t = time_steps.float().to(device).flatten()
        mean,std = model(t,data)
        loss = nll(mean,std,batch.float().to(device),lat,var_coeff)
        if torch.isnan(loss) : 
            print("Quitting due to Nan loss")
            quit()
        print("Val Loss for batch is ",loss.item())
        val_loss = val_loss + loss.item()

    print("|Iter ",epoch," | Total Val Loss ", val_loss,"|")


    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        torch.save(model,str(cwd) + "/Models/" + "ClimODE_global_"+args.solver+"_"+str(args.spectral)+"_model_" + str(epoch) + ".pt")


