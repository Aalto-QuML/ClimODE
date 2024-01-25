import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
#import warnings
from model_utils import *
from utils import *
from torchdiffeq import odeint as odeint


class Optim_velocity(nn.Module):
    def __init__(self,num_years,H,W):
        super(Optim_velocity,self).__init__()
        self.v_x = torch.nn.Parameter(torch.randn(num_years,1,5,H,W))
        self.v_y = torch.nn.Parameter(torch.randn(num_years,1,5,H,W))

    def forward(self,data):
        u_y = torch.gradient(data,dim=3)[0] # (H,W) --> (y,x)
        u_x = torch.gradient(data,dim=4)[0]
        adv = self.v_x*u_x + self.v_y*u_y + data*(torch.gradient(self.v_y,dim=3)[0] + torch.gradient(self.v_x,dim=4)[0])
        out = adv
        return out,self.v_x,self.v_y 


class Climate_ResNet_2D(nn.Module): 
    
    def __init__(self,num_channels,layers,hidden_size):
        super().__init__()
        layers_cnn = []
        activation_fns = []
        self.block = ResidualBlock
        self.inplanes = num_channels

        for idx in range(len(layers)):
            if idx ==0:
               layers_cnn.append(self.make_layer(self.block,num_channels,hidden_size[idx],layers[idx]))
            else:
                layers_cnn.append(self.make_layer(self.block,hidden_size[idx-1],hidden_size[idx],layers[idx]))
        
        self.layer_cnn = nn.ModuleList(layers_cnn)
        self.activation_cnn = nn.ModuleList(activation_fns)

    def make_layer(self,block,in_channels,out_channels,reps):
        layers = []
        layers.append(block(in_channels,out_channels))
        self.inplanes = out_channels
        for i in range(1, reps):  
              layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self,data):
        dx_final = data.float()
        for l,layer in enumerate(self.layer_cnn):
            dx_final = layer(dx_final)

                
        return dx_final



class ClimODE_uncertain(nn.Module): 
    
    def __init__(self,num_channels,const_channels,out_types,method,use_att,use_err,use_pos):
        super().__init__()
        self.layers = [5,3,2]
        self.hidden = [128,64,2*out_types]
        input_channels = 30 + out_types*int(use_pos) + 34*(1-int(use_pos))
        self.vel_f = Climate_ResNet_2D(input_channels,self.layers,self.hidden)

        if use_att: 
            self.vel_att = Self_attn_conv(input_channels,2*out_types)
            self.gamma = nn.Parameter(torch.tensor([0.1]))

        self.scales = num_channels
        self.const_channel = const_channels
        
        self.out_ch = out_types
        self.past_samples = 0
        self.const_info = 0
        self.lat_map = 0
        self.lon_map = 0
        self.elev = 0
        self.pos_emb = 0
        self.elev_info_grad_x = 0
        self.elev_info_grad_y = 0
        self.method = method
        err_in =  9 + out_types*int(use_pos) + 34*(1-int(use_pos))
        if use_err: self.noise_net = Climate_ResNet_2D(err_in,[3,2,2],[128,64,2*out_types])
        if use_pos: self.pos_enc = Climate_ResNet_2D(4,[2,1,1],[32,16,out_types])
        self.att = use_att
        self.err = use_err
        self.pos = use_pos
        self.pos_feat = 0
        self.lsm =0 
        self.oro =0 


    def update_param(self, params):
        self.past_samples = params[0]
        self.const_info = params[1]
        self.lat_map = params[2]
        self.lon_map = params[3]


    def pde(self,t,vs):

        ds = vs[:,-self.out_ch:,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()
        v = vs[:,:2*self.out_ch,:,:].float().view(-1,2*self.out_ch,vs.shape[2],vs.shape[3]).float()
        t_emb = ((t*100)%24).view(1,1,1,1).expand(ds.shape[0],1,ds.shape[2],ds.shape[3])
        sin_t_emb = torch.sin(torch.pi*t_emb/12 - torch.pi/2)
        cos_t_emb = torch.cos(torch.pi*t_emb/12 - torch.pi/2)
        
        sin_seas_emb = torch.sin(torch.pi*t_emb/(12*365) - torch.pi/2)
        cos_seas_emb = torch.cos(torch.pi*t_emb/(12*365) - torch.pi/2)

        day_emb = torch.cat([sin_t_emb,cos_t_emb],dim=1)
        seas_emb = torch.cat([sin_seas_emb,cos_seas_emb],dim=1)
        
        ds_grad_x = torch.gradient(ds,dim=3)[0]
        ds_grad_y = torch.gradient(ds,dim=2)[0]
        nabla_u = torch.cat([ds_grad_x,ds_grad_y],dim=1)

        if self.pos:
            comb_rep = torch.cat([t_emb/24,day_emb,seas_emb,nabla_u,v,ds,self.pos_feat],dim=1)
        else:
            cos_lat_map,sin_lat_map = torch.cos(self.new_lat_map),torch.sin(self.new_lat_map)
            cos_lon_map,sin_lon_map = torch.cos(self.new_lon_map),torch.sin(self.new_lon_map)
            t_cyc_emb = torch.cat([day_emb,seas_emb],dim=1)
            pos_feats = torch.cat([cos_lat_map,cos_lon_map,sin_lat_map,sin_lon_map,sin_lat_map*cos_lon_map,sin_lat_map*sin_lon_map],dim=1)
            pos_time_ft = self.get_time_pos_embedding(t_cyc_emb,pos_feats)
            comb_rep = torch.cat([t_emb/24,day_emb,seas_emb,nabla_u,v,ds,self.new_lat_map,self.new_lon_map,self.lsm,self.oro,pos_feats,pos_time_ft],dim=1)

        if self.att: dv = self.vel_f(comb_rep) + self.gamma*self.vel_att(comb_rep)
        else: dv = self.vel_f(comb_rep)
        v_x = v[:,:self.out_ch,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()
        v_y = v[:,-self.out_ch:,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()

        adv1 = v_x*ds_grad_x + v_y*ds_grad_y
        adv2 = ds*(torch.gradient(v_x,dim=3)[0] + torch.gradient(v_y,dim=2)[0] )

        ds = adv1 + adv2 
        dvs = torch.cat([dv,ds],1)
        return dvs


    def get_time_pos_embedding(self,time_feats,pos_feats):
        for idx in range(time_feats.shape[1]):
            tf = time_feats[:,idx].unsqueeze(dim=1)*pos_feats
            if idx == 0:
                final_out = tf
            else:
                final_out = torch.cat([final_out,tf],dim=1)

        return final_out

    def noise_net_contrib(self,t,pos_enc,s_final,noise_net,H,W):

        t_emb = (t%24).view(-1,1,1,1,1)
        sin_t_emb = torch.sin(torch.pi*t_emb/12 - torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)
        cos_t_emb = torch.cos(torch.pi*t_emb/12 - torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)
        
        sin_seas_emb = torch.sin(torch.pi*t_emb/(12*365)- torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)
        cos_seas_emb = torch.cos(torch.pi*t_emb/(12*365)- torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)

        pos_enc = pos_enc.expand(len(s_final),s_final.shape[1],-1,H,W).flatten(start_dim=0,end_dim=1)
        t_cyc_emb = torch.cat([sin_t_emb,cos_t_emb,sin_seas_emb,cos_seas_emb],dim=2).flatten(start_dim=0,end_dim=1)

        pos_time_ft = self.get_time_pos_embedding(t_cyc_emb,pos_enc[:,2:-2])

        comb_rep = torch.cat([t_cyc_emb,s_final.flatten(start_dim=0,end_dim=1),pos_enc,pos_time_ft],dim=1)

        final_out = noise_net(comb_rep).view(len(t),-1,2*self.out_ch,H,W)

        mean = s_final + final_out[:,:,:self.out_ch]
        std = nn.Softplus()(final_out[:,:,self.out_ch:])
        
        return mean,std

    def forward(self,T,data,atol=0.1,rtol=0.1):
        H,W = self.past_samples.shape[2],self.past_samples.shape[3]
        final_data = torch.cat([self.past_samples ,data.float().view(-1,self.out_ch,H,W)],1)
        init_time = T[0].item()*6
        final_time = T[-1].item()*6
        steps_val = final_time - init_time
        
        #breakpoint()

        if self.pos:
            lat_map = self.lat_map.unsqueeze(dim=0)*torch.pi/180
            lon_map = self.lon_map.unsqueeze(dim=0)*torch.pi/180
            pos_rep = torch.cat([lat_map.unsqueeze(dim=0),lon_map.unsqueeze(dim=0),self.const_info],dim=1)
            self.pos_feat = self.pos_enc(pos_rep).expand(data.shape[0],-1,data.shape[3],data.shape[4])
            final_pos_enc = self.pos_feat
        
        else:
            self.oro,self.lsm = self.const_info[0,0],self.const_info[0,1]
            self.lsm = self.lsm.unsqueeze(dim=0).expand(data.shape[0],-1,data.shape[3],data.shape[4])
            self.oro  = F.normalize(self.const_info[0,0]).unsqueeze(dim=0).expand(data.shape[0],-1,data.shape[3],data.shape[4])
            self.new_lat_map = self.lat_map.expand(data.shape[0],1,data.shape[3],data.shape[4])*torch.pi/180 # Converting to radians
            self.new_lon_map = self.lon_map.expand(data.shape[0],1,data.shape[3],data.shape[4])*torch.pi/180
            cos_lat_map,sin_lat_map = torch.cos(self.new_lat_map),torch.sin(self.new_lat_map)
            cos_lon_map,sin_lon_map = torch.cos(self.new_lon_map),torch.sin(self.new_lon_map)
            pos_feats = torch.cat([cos_lat_map,cos_lon_map,sin_lat_map,sin_lon_map,sin_lat_map*cos_lon_map,sin_lat_map*sin_lon_map],dim=1)
            final_pos_enc = torch.cat([self.new_lat_map,self.new_lon_map,pos_feats,self.lsm,self.oro],dim=1)


        new_time_steps = torch.linspace(init_time,final_time,steps=int(steps_val)+1).to(data.device)
        t = 0.01*new_time_steps.float().to(data.device).flatten().float()
        pde_rhs  = lambda t,vs: self.pde(t,vs) # make the ODE forward function
        final_result = odeint(pde_rhs,final_data,t,method=self.method,atol=atol,rtol=rtol)
        s_final = final_result[:,:,-self.out_ch:,:,:].view(len(t),-1,self.out_ch,H,W)

        if self.err:
            mean,std = self.noise_net_contrib(T,final_pos_enc,s_final[0:len(s_final):6],self.noise_net,H,W)

        else:
            s_final = s_final[0:len(s_final):6]

        return mean,std



class ClimODE_uncertain_region(nn.Module): 
    
    def __init__(self,num_channels,const_channels,out_types,method,use_att,use_err,use_pos):
        super().__init__()
        self.layers = [5,3,2]
        self.hidden = [128,64,2*out_types]
        input_channels = 30 + out_types*int(use_pos) + 34*(1-int(use_pos))
        self.vel_f = Climate_ResNet_2D(input_channels,self.layers,self.hidden)

        if use_att: 
            self.vel_att = Self_attn_conv_reg(input_channels,2*out_types)
            self.gamma = nn.Parameter(torch.tensor([0.1]))

        self.scales = num_channels
        self.const_channel = const_channels
        
        self.out_ch = out_types
        self.past_samples = 0
        self.const_info = 0
        self.lat_map = 0
        self.lon_map = 0
        self.elev = 0
        self.pos_emb = 0
        self.elev_info_grad_x = 0
        self.elev_info_grad_y = 0
        self.method = method
        err_in =  9 + out_types*int(use_pos) + 34*(1-int(use_pos))
        if use_err: self.noise_net = Climate_ResNet_2D(err_in,[3,2,2],[128,64,2*out_types])
        if use_pos: self.pos_enc = Climate_ResNet_2D(4,[2,1,1],[32,16,out_types])
        self.att = use_att
        self.err = use_err
        self.pos = use_pos
        self.pos_feat = 0
        self.lsm =0 
        self.oro =0 

    def update_param(self, params):
        self.past_samples = params[0]
        self.const_info = params[1]
        self.lat_map = params[2]
        self.lon_map = params[3]

    def pde(self,t,vs):

        ds = vs[:,-self.out_ch:,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()
        v = vs[:,:2*self.out_ch,:,:].float().view(-1,2*self.out_ch,vs.shape[2],vs.shape[3]).float()
        t_emb = ((t*100)%24).view(1,1,1,1).expand(ds.shape[0],1,ds.shape[2],ds.shape[3])
        sin_t_emb = torch.sin(torch.pi*t_emb/12 - torch.pi/2)
        cos_t_emb = torch.cos(torch.pi*t_emb/12 - torch.pi/2)
        
        sin_seas_emb = torch.sin(torch.pi*t_emb/(12*365) - torch.pi/2)
        cos_seas_emb = torch.cos(torch.pi*t_emb/(12*365) - torch.pi/2)

        day_emb = torch.cat([sin_t_emb,cos_t_emb],dim=1)
        seas_emb = torch.cat([sin_seas_emb,cos_seas_emb],dim=1)
        
        ds_grad_x = torch.gradient(ds,dim=3)[0]
        ds_grad_y = torch.gradient(ds,dim=2)[0]
        nabla_u = torch.cat([ds_grad_x,ds_grad_y],dim=1)

        if self.pos:
            comb_rep = torch.cat([t_emb/24,day_emb,seas_emb,nabla_u,v,ds,self.pos_feat],dim=1)
        else:
            cos_lat_map,sin_lat_map = torch.cos(self.new_lat_map),torch.sin(self.new_lat_map)
            cos_lon_map,sin_lon_map = torch.cos(self.new_lon_map),torch.sin(self.new_lon_map)
            t_cyc_emb = torch.cat([day_emb,seas_emb],dim=1)
            pos_feats = torch.cat([cos_lat_map,cos_lon_map,sin_lat_map,sin_lon_map,sin_lat_map*cos_lon_map,sin_lat_map*sin_lon_map],dim=1)
            pos_time_ft = self.get_time_pos_embedding(t_cyc_emb,pos_feats)
            comb_rep = torch.cat([t_emb/24,day_emb,seas_emb,nabla_u,v,ds,self.new_lat_map,self.new_lon_map,self.lsm,self.oro,pos_feats,pos_time_ft],dim=1)

        if self.att: dv = self.vel_f(comb_rep) + self.gamma*self.vel_att(comb_rep)
        else: dv = self.vel_f(comb_rep)
        v_x = v[:,:self.out_ch,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()
        v_y = v[:,-self.out_ch:,:,:].float().view(-1,self.out_ch,vs.shape[2],vs.shape[3]).float()

        adv1 = v_x*ds_grad_x + v_y*ds_grad_y
        adv2 = ds*(torch.gradient(v_x,dim=3)[0] + torch.gradient(v_y,dim=2)[0] )

        ds = adv1 + adv2 
        dvs = torch.cat([dv,ds],1)
        return dvs
    

    def get_time_pos_embedding(self,time_feats,pos_feats):
        for idx in range(time_feats.shape[1]):
            tf = time_feats[:,idx].unsqueeze(dim=1)*pos_feats
            if idx == 0:
                final_out = tf
            else:
                final_out = torch.cat([final_out,tf],dim=1)

        return final_out

    def noise_net_contrib(self,t,pos_enc,s_final,noise_net,H,W):

        t_emb = (t%24).view(-1,1,1,1,1)
        sin_t_emb = torch.sin(torch.pi*t_emb/12 - torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)
        cos_t_emb = torch.cos(torch.pi*t_emb/12 - torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)
        
        sin_seas_emb = torch.sin(torch.pi*t_emb/(12*365)- torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)
        cos_seas_emb = torch.cos(torch.pi*t_emb/(12*365)- torch.pi/2).expand(len(s_final),s_final.shape[1],1,H,W)

        pos_enc = pos_enc.expand(len(s_final),s_final.shape[1],-1,H,W).flatten(start_dim=0,end_dim=1)
        t_cyc_emb = torch.cat([sin_t_emb,cos_t_emb,sin_seas_emb,cos_seas_emb],dim=2).flatten(start_dim=0,end_dim=1)

        pos_time_ft = self.get_time_pos_embedding(t_cyc_emb,pos_enc[:,2:-2])

        comb_rep = torch.cat([t_cyc_emb,s_final.flatten(start_dim=0,end_dim=1),pos_enc,pos_time_ft],dim=1)

        final_out = noise_net(comb_rep).view(len(t),-1,2*self.out_ch,H,W)

        mean = s_final + final_out[:,:,:self.out_ch]
        std = nn.Softplus()(final_out[:,:,self.out_ch:])
        
        return mean,std


    def forward(self,T,data,atol=0.1,rtol=0.1):
        H,W = self.past_samples.shape[2],self.past_samples.shape[3]
        final_data = torch.cat([self.past_samples ,data.float().view(-1,self.out_ch,H,W)],1)
        init_time = T[0].item()*6
        final_time = T[-1].item()*6
        steps_val = final_time - init_time
        
        #breakpoint()

        if self.pos:
            lat_map = self.lat_map.unsqueeze(dim=0)*torch.pi/180
            lon_map = self.lon_map.unsqueeze(dim=0)*torch.pi/180
            pos_rep = torch.cat([lat_map.unsqueeze(dim=0),lon_map.unsqueeze(dim=0),self.const_info],dim=1)
            self.pos_feat = self.pos_enc(pos_rep).expand(data.shape[0],-1,data.shape[3],data.shape[4])
            final_pos_enc = self.pos_feat
        
        else:
            self.oro,self.lsm = self.const_info[0,0],self.const_info[0,1]
            self.lsm = self.lsm.unsqueeze(dim=0).expand(data.shape[0],-1,data.shape[3],data.shape[4])
            self.oro  = F.normalize(self.const_info[0,0]).unsqueeze(dim=0).expand(data.shape[0],-1,data.shape[3],data.shape[4])
            self.new_lat_map = self.lat_map.expand(data.shape[0],1,data.shape[3],data.shape[4])*torch.pi/180 # Converting to radians
            self.new_lon_map = self.lon_map.expand(data.shape[0],1,data.shape[3],data.shape[4])*torch.pi/180
            cos_lat_map,sin_lat_map = torch.cos(self.new_lat_map),torch.sin(self.new_lat_map)
            cos_lon_map,sin_lon_map = torch.cos(self.new_lon_map),torch.sin(self.new_lon_map)
            pos_feats = torch.cat([cos_lat_map,cos_lon_map,sin_lat_map,sin_lon_map,sin_lat_map*cos_lon_map,sin_lat_map*sin_lon_map],dim=1)
            final_pos_enc = torch.cat([self.new_lat_map,self.new_lon_map,pos_feats,self.lsm,self.oro],dim=1)


        new_time_steps = torch.linspace(init_time,final_time,steps=int(steps_val)+1).to(data.device)
        t = 0.01*new_time_steps.float().to(data.device).flatten().float()
        pde_rhs  = lambda t,vs: self.pde(t,vs) # make the ODE forward function
        final_result = odeint(pde_rhs,final_data,t,method=self.method,atol=atol,rtol=rtol)
        s_final = final_result[:,:,-self.out_ch:,:,:].view(len(t),-1,self.out_ch,H,W)

        if self.err:
            mean,std = self.noise_net_contrib(T,final_pos_enc,s_final[0:len(s_final):6],self.noise_net,H,W)

        else:
            s_final = s_final[0:len(s_final):6]

        return mean,std
