
#####  Python libraries import #####

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math as math
import random as rand
import sys
import pandas as pd
import torch
import igraph as ig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#####  Simulation parameters #####

d=2 #system dimension
packing_fraction=0.07 
N=10000 # population size
mu= 1 #mobility
Frep=40 # repulsive strength
Fadh= 7 # adhesive strength
Req= 1.1 # equilibrium diameter
R0= 1.6 # interaction range 
f1=0.5 # Fraction of v1-particles type in the mix
n1=np.int(f1*N) # Number of v1-particles
v1,v2=8,8 # particles motility
aux1,aux2=torch.ones(n1)*v1,torch.ones(N-n1)*v2
v0=torch.cat((aux1,aux2),dim=-1)
n = torch.rand(N,d)-0.5  # initial particles auto-propulasion direction
nabs=torch.sqrt(torch.sum(n**2,1))
n=torch.div(n,nabs[:,None])
noise=10 # noise intensity
tau=5 # characteristic time for the polarization to align in the scattering direction defined by v=dr/dt
tf= 100 # simulation time
dt= 0.01 #timesteps

box_size= 28 # the simulation space is subdivided into boxes to save calculation time
while box_size%4 != 0:
  print("Box size should be integer multiple of 4")
  print("Enter new box_size value?")
  inpu = sys.stdin.readline()
  box_size = np.int(inpu.split()[0]) 
Lx=int(math.sqrt(N*Req/2*Req/2*math.pi/packing_fraction))
print('Lx=',Lx)
if Lx < box_size:
  Lx=Lx-Lx%4
  box_size=int(Lx/2)
else:
  Lx=Lx-Lx%box_size
  if Lx == box_size:
    box_size=int(Lx/2)
Ly=Lx
nx,ny=int(Lx/box_size),int(Ly/box_size)
nt=nx*ny # box number
print("rho=%f, Lx=%d, nx=%d, ny=%d, nt=%d"%(N/(Lx*Ly),Lx,nx,ny,nt))
L=torch.tensor([Lx,Ly])
X = torch.rand(N,d) 
X=X*(L)  # initial particles position
ll= Lx

steps=tf/dt # number of simulation steps
N_fig=100 # number of snapshots of the system saved during the simulation
exit_fig=int(steps/N_fig)
N_op=100# number of order parameter measurements during the simulation
exit_op=int(steps/N_op)
intt=0
sizes=5 # particles size for plotting


##### Torch molecular simulation function #####

def bc_pos(X): # particles position peridocity
        return torch.remainder(X,ll)#
def bc_diff(D): # particles distances periodicity
        return torch.remainder(D-(ll/2),ll)-(ll/2) # same thing
def distmat_square_inbox(X): # pairwise distances within a box
        D = torch.sum((X[:,None,:]-X[None,:,:])**2,axis=2)
        D = torch.where(D < 0.00001*torch.ones(1,device=device), torch.ones(1,device=device),D)
        return D
def distmat_square_interbox(X,Y):  # pairwise distances between bo
    D = torch.sum(bc_diff(X[:,None,:]-Y[None,:,:])**2,axis=2)
    return D
def distmat_square(X): # pairwise distances between every particles
        return torch.sum(bc_diff(X[:,None,:]-X[None,:,:])**2,axis=2)
def force_mod(R,zero_tensor): # interaction forces calculation
        R=torch.sqrt(R)
        frep=-Frep*(1/Req-1/R)
        frep=torch.where(R<Req,frep,zero_tensor)
        fadh=-Fadh*(1-Req/R)/(R0-Req)
        fadh=torch.where(R>Req,fadh,zero_tensor)
        fadh=torch.where(R<R0,fadh,zero_tensor)
        force=fadh+frep
        return  force
def force_field_inbox(X,D,zero_tensor): # force field in the focal box
        FF=torch.sum(force_mod(D,zero_tensor)[:,:,None]*(X[:,None,:]-X[None,:,:]),axis=1)
        return FF 
def force_field_interbox(X,Y,D,zero_tensor): # force field with the neighbouring boxes
        if len(X)==0 or len(Y) == 0:
                return torch.zeros(2,device=device),torch.zeros(2,device=device)
        else: 
                force = force_mod(D,zero_tensor)
                FF_target_box = torch.sum(force[:,:,None]*bc_diff((X[:,None,:]-Y[None,:,:])),axis=1)
                FF_reaction = -torch.sum(force[:,:,None]*bc_diff((X[:,None,:]-Y[None,:,:])),axis=0)
                return FF_target_box,FF_reaction
def autovel(dX,n):
        theta=torch.atan2(dX[:,1],dX[:,0])
        dXabs=torch.sqrt(torch.sum(dX**2,1))
        dX_norm=torch.div(dX,dXabs[:,None])*0.9999999
        dtheta=torch.arcsin((n[:,0]*dX_norm[:,1]-n[:,1]*dX_norm[:,0]))*dt/tau
        rnd=noise*(2*math.pi*(torch.rand(len(dX),1,device=device)-0.5))*np.sqrt(dt)
        theta+=dtheta+rnd[:,0]
        n[:,0]=torch.cos(theta)
        n[:,1]=torch.sin(theta)
        return n
def boite(X,box_size,nx,nt,N,delta):
        box_size2=box_size/4
        box=(X[:,0]/box_size).to(dtype=int)+nx*(X[:,1]/box_size).to(dtype=int)
        box2=((X[:,0])/box_size2).to(dtype=int)+4*nx*((X[:,1])/box_size2).to(dtype=int) #finer box division to incorporate less neighborhood particles
        box=list(map(int,box))
        box2=list(map(int,box2))
        # list of particles in the focal box
        box_part_list=list( [] for i in range(nt))
        box_part_list2=list( [] for i in range(16*nt))
        #counting the particles on each box and composing the box list of particles
        for i in range(N):
                box_part_list[box[i]].append(i)
                box_part_list2[box2[i]].append(i)   
        #constructing the lists of particles in the neighbor boxes
        neighbox_list=list( [] for i in range(nt))
        for i in range(nt):
                kx,ky=i%nx,int(i/nx) #i%nx: X position of the box, int(i/nx): Y-position of the box
                zz1=(4*ky*4*nx+4*(kx+1)%(4*nx))%(16*nt)      
                zz2=(4*ky*4*nx+4*nx+4*(kx+1)%(4*nx))%(16*nt)
                zz3=(4*ky*4*nx+8*nx+4*(kx+1)%(4*nx))%(16*nt)           #boxes on the right
                zz4=(4*ky*4*nx+12*nx+4*(kx+1)%(4*nx))%(16*nt)
                zz5=(16*(ky+1)*nx+(4*kx)%(4*nx))%(16*nt)     
                zz6=(16*(ky+1)*nx+(4*kx+1)%(4*nx))%(16*nt)
                zz7=(16*(ky+1)*nx+(4*kx+2)%(4*nx))%(16*nt)           #lower boxes
                zz8=(16*(ky+1)*nx+(4*kx+3)%(4*nx))%(16*nt)
                zz9=(16*(ky+1)*nx+(4*kx-1)%(4*nx))%(16*nt)
                zz10=(16*(ky+1)*nx+(4*kx+4)%(4*nx))%(16*nt)
                neighbox_list[i].extend(box_part_list2[int(zz1)])
                neighbox_list[i].extend(box_part_list2[int(zz2)])
                neighbox_list[i].extend(box_part_list2[int(zz3)])
                neighbox_list[i].extend(box_part_list2[int(zz4)])
                neighbox_list[i].extend(box_part_list2[int(zz5)])
                neighbox_list[i].extend(box_part_list2[int(zz6)])
                neighbox_list[i].extend(box_part_list2[int(zz7)])
                neighbox_list[i].extend(box_part_list2[int(zz8)])
                neighbox_list[i].extend(box_part_list2[int(zz9)])
                neighbox_list[i].extend(box_part_list2[int(zz10)])
        return box_part_list,neighbox_list
def OP_dynamics(Coords):
  Pairwise_dist=distmat_square(Coords)
  interaction=torch.where(torch.sqrt(Pairwise_dist) < R0, 1*torch.ones(1,device=device), 0*torch.ones(1,device=device)) # 2 particles are considered connected (=1) if their pairwise distance at the end of the simulation (<R0))
  Interaction=interaction.to("cpu") # torch tensor -> numpy array
  Interaction=Interaction.numpy()
  node_names = [i for i in range(N)] #node names= particles id = i or j index
  Interaction=pd.DataFrame(Interaction,index=node_names, columns=node_names) # numpy array -> pd.dataframe
  Values = Interaction.values 
  g = ig.Graph.Adjacency((Values > 0).tolist(),diag=False) # build the graph from the adjency matrix = "Interaction", diag=False to discard the diagonal
  g.vs['label'] = node_names #name the nodes
  gg=g.clusters() # identify the clusters = connected components of the graph
  Agg_List=[gg[i] for i in range(len(gg)) if len(gg[i])>4] # clusters whose size is lower than threshold2 are discarded
  Agg_List=np.hstack(Agg_List) # List of clustered particles
  AGG_STAT=0*torch.ones(N,device=device)
  AGG_STAT[Agg_List]=torch.ones(1,device=device) 
  # Aggregated fraction
  AggFract1=torch.sum(AGG_STAT[:n1])/n1
  AggFract2=torch.sum(AGG_STAT[n1:])/(N-n1)
  AggFract=torch.sum(AGG_STAT)/N
  # Aggregates size and aggregate number 
  # list of clusters size
  Sagg_thr=[gg.size(i) for i in range(len(gg)) if gg.size(i)>4] # the clusters composed of less than 5 particles are not considered as aggregates
  Mean_sagg=np.mean(Sagg_thr)
  Nagg_thr=len(Sagg_thr)# number of aggregatesxx1[i]
  # aggregates composition 
  AggComp_v1=[sum([1 for k in gg[i] if k < int(f1*N)])/len(gg[i]) for i in range(len(gg)) if len(gg[i])>4] # aggregates composition = number of v1-particles/ aggregate size
  Var_AggComp=np.var(AggComp_v1) # variance in aggregates composition 
  Mean_AggComp=np.mean(AggComp_v1) # mean aggregates composition
  return float(Nagg_thr),float(Mean_sagg),float(AggFract)


##### Initialization #####

intt=0
#defining torch device, tensors and sending tensor to devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delta=torch.tensor([0.001]).to(device)
v0 = v0.to(device)
X = X.to(device)
L = L.to(device)
n = n.to(device)
box_part_list,neighbox_list=boite(X,box_size,nx,nt,N,delta)
t=0
# order parameter dynamics lists
Nagg,Sagg,Aggf1,Aggf2,Aggf,AggComp=[],[],[],[],[],[]

def images(figindex,sizes):
    plt.figure(figsize=(8,8))
    plt.axis([0,Lx,0,Ly])
    plt.axes().set_aspect(1.0)
    X1=X[torch.where(v0==v1)]
    X2=X[torch.where(v0==v2)]
    x1=[np.array(i.cpu()) for i in X1]
    xx1=[float(i[0]) for i in x1]
    xy1=[float(i[1]) for i in x1]
    x2=[np.array(i.cpu()) for i in X2]
    xx2=[float(i[0]) for i in x2]
    xy2=[float(i[1]) for i in x2] 
    plt.scatter(xx1,xy1,s=sizes,c='gray',alpha=0.5)
    plt.scatter(xx2,xy2,s=sizes,c='gray',alpha=0.5)
    name=str(figindex)
    fig = plt.gcf()
    plt.rc("savefig",dpi=200)
    fig.savefig(name,bbox_inches='tight')
    plt.close()
    
##### System evolution #####

while t < tf :
        F=torch.zeros(N,2,device=device) #zero forces
        #loop over boxes
        for i in range(nt):
                X_box =X[box_part_list[i]]  #position of particles in box i
                num_box=len(X_box)     #number of particles in box i
                if num_box != 0 :
                        zero_tensor=torch.zeros(num_box,num_box,device=device)  #used to calculate forces
                        D_inbox = distmat_square_inbox(X_box)       #distance among particles in box i
                        F_box=force_field_inbox(X_box,D_inbox,zero_tensor)    #forces among particles in box i
                        F[box_part_list[i]]+=F_box      #adding to the global force tensor
                        X_box_neigh = X[neighbox_list[i]]    #position of particles in neighbor boxes
                        zero_tensor=torch.zeros(len(box_part_list[i]),len(neighbox_list[i]),device=device)
                        D_interbox=distmat_square_interbox(X_box,X_box_neigh)    #distance between particles in box i and particles in the neighboring boxese
                        FF_target_box,FF_reaction=force_field_interbox(X_box,X_box_neigh,D_interbox,zero_tensor) #forces among particles in box i and in neighboring boxes, also reaction force in the neighboring particles is calculated
                        F[box_part_list[i]]+=FF_target_box  #add forces produced in the interaction with part in neighboring boxes
                        F[neighbox_list[i]]+=FF_reaction
        #evolve all positions
        dX = mu*F*dt + v0[:,None]*n*dt
        n=autovel(dX,n)
        X+=dX
        t+=dt
        intt+=1
        X=bc_pos(X) # periodicity
        X=bc_pos(X)
        if intt%10 == 0: # estimate new particles boxes every 10 time steps
                box_part_list,neighbox_list=boite(X,box_size,nx,nt,N,delta)
        if(intt%exit_fig==0):
          #Images of instantaneous particle positions. the temporal resolution is set by "exit_fig"
          images(intt,sizes)
        if(intt%exit_op==0):
          #OPs recording
          nagg,sagg,aggf=OP_dynamics(X)
          Nagg.append(nagg)
          Sagg.append(sagg)
          #Aggf1.append(aggf1)
          #Aggf2.append(aggf2)
          #Aggf.append(aggf)
          #AggComp.append(aggComp)
D_fin=distmat_square(X)


##### Groups identification #####

threshold=4
interaction=torch.where(torch.sqrt(D_fin) < R0, 1*torch.ones(1,device=device), 0*torch.ones(1,device=device)) # 2 particles are considered connected (=1) if their pairwise distance at the end of the simulation (<R0))
interaction=interaction.fill_diagonal_(0) # make sure that the diagonal is filled with 0
Interaction=interaction.to("cpu") # torch tensor -> numpy array
Interaction=Interaction.numpy()
node_names = [i for i in range(N)] #node names= particles id = i or j index
Interaction=pd.DataFrame(Interaction,index=node_names, columns=node_names) # numpy array -> pd.dataframe
Values = Interaction.values 
g = ig.Graph.Adjacency((Values > 0).tolist(),diag=False) # build the graph from the adjency matrix = "Interaction", diag=False to discard the diagonal
g.vs['label'] = node_names #name the nodes
gg=g.clusters() # identify the clusters = connected components of the graph
Agg_List=[gg[i] for i in range(len(gg)) if len(gg[i])>threshold] # clusters whose size is lower than threshold are discarded
Agg_List=np.hstack(Agg_List) # List of clustered particles
Agg_STAT=0*torch.ones(N,device=device)
Agg_STAT[Agg_List]=torch.ones(1,device=device) # 1 if a particle is clustered



##### Final order parameters estimation #####

#Aggregated fraction
AggFract1=torch.sum(Agg_STAT[:n1])/n1
AggFract2=torch.sum(Agg_STAT[n1:])/(N-n1)
AggFract=torch.sum(Agg_STAT)/N
print("aggregated particles fraction = "+str(float(AggFract)))
print("aggregated type 1 particles fraction = "+str(float(AggFract1)))
print("aggregated type 2 particles fraction = "+str(float(AggFract2)))

# Aggregates size and aggregate number 
Nagg=len(gg) # gg= list of clusters, isolated vertices= isolated clusters are considered as size 1 clusters
Sagg=[gg.size(i) for i in range(Nagg)] # list of clusters size
Sagg_thr=[s for s in Sagg if s>4] # the clusters composed of less than 5 particles are not considered as aggregates
Nagg_thr=len(Sagg_thr)# number of aggregates
print('Nagg=', Nagg_thr)
print('Mean Agg Size=',np.mean(Sagg_thr))
Size_distrib=[gg.size(i) for i in range(len(gg))] # aggregates size distribution

# aggregates composition 
AggComp_v1=[sum([1 for k in gg[i] if k < int(f1*N)])/len(gg[i]) for i in range(len(gg)) if len(gg[i])>threshold] # aggregates composition = number of v1-particles/ aggregate size
Var_AggComp=np.var(AggComp_v1) # variance in aggregates composition
Mean_AggComp=np.mean(AggComp_v1) # mean aggregates composition
print('Mean Agg Comp=',Mean_AggComp)
print('Var Agg Comp=', Var_AggComp)
# the aggregates composition variance is normalised by the maximal variance that could be obtained given the number of aggregates and the number of aggregated particles from the two pop
# namely when the v1-particles and v2-particles are seggregated in the different aggregates

Agg_mean_size=np.mean(Sagg_thr)
N_agg_1=int(torch.sum(Agg_STAT[:n1])/Agg_mean_size)
N_agg_2=int(torch.sum(Agg_STAT[n1:])/Agg_mean_size)
Sorted_agg_comp=[0 for i in range(int(N_agg_1+N_agg_2))]
for j in range(int(N_agg_1+N_agg_2)):
  if j<=N_agg_1:
    Sorted_agg_comp[j]=1 
norm_var=Var_AggComp/np.var(Sorted_agg_comp)
print('Var Agg Comp (standardized)=', norm_var)

# bias in v1-particles aggregates composition

Bias=torch.sum(Agg_STAT[:n1])/(torch.sum(Agg_STAT[:n1])+torch.sum(Agg_STAT[n1:]))-f1
print('Bias='+str(Bias))

# particles connectivity
                               
# v1-particles
# total number of neighbors
v1_part_degree_tot=np.mean(Interaction.sum(1)[:int(f1*N)][Interaction.iloc[:int(f1*N)].sum(1)!=0]) 
# number of neighbors from the same type
v1_part_degree_self=np.mean(Interaction.iloc[:int(f1*N),:int(f1*N)].sum(1)[Interaction.iloc[:int(f1*N)].sum(1)!=0])
# number of neighbors from the other type
v1_part_degree_nonself=np.mean(Interaction.iloc[:int(f1*N),int(f1*N):].sum(1)[Interaction.iloc[:int(f1*N):].sum(1)!=0])
print('v1-particles mean connectivity='+str(v1_part_degree_tot))
# v2-particles
# total number of neighbors
v2_part_degree_tot=np.mean(Interaction.sum(axis=1)[int(f1*N):][Interaction.iloc[int(f1*N):].sum(axis=1)!=0]) 
# number of neighbors from the same type
v2_part_degree_self=np.mean(Interaction.iloc[int(f1*N):,int(f1*N):].sum(1)[Interaction.iloc[int(f1*N):].sum(1)!=0])
# number of neighbors from the other type
v2_part_degree_nonself=np.mean(Interaction.iloc[int(f1*N):,:int(f1*N)].sum(1)[Interaction.iloc[int(f1*N):].sum(1)!=0])
print('v2-particles mean connectivity='+str(v2_part_degree_tot))



