from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.animation as animation
import pandas as pd

import os 
os.chdir("../simulationData")



# Simulation Data
CirName = "1-1-1-1-1"
Data1 = np.load("3dFDTD_Circuit"+CirName+".npz")
Jx_xz    = Data1["Jx_xz"]
Jx_yz    = Data1["Jx_yz"]
dt1      = Data1["dt"]

dx  = Data1["dx"]
dy  = Data1["dy"]
dz  = Data1["dz"]
x = np.linspace(0,dx*Jx_xz.shape[0],Jx_xz.shape[0])

plt.rcParams['font.family'] ='Helvetica'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 18 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

z_Si = 32
z_Gi = 38
z_Go = 42

fig=plt.figure(figsize=(7,5))

nFrame = 300
rate = 3
vmax=Jx_xz.max()/5
vmin=-Jx_xz.max()/5

def update(i):
    plt.clf()
    # time = rate*i+14000
    time = rate*i
    # t = dt*time
    print(time)
    ### Plot ###
    plt.plot(x,Jx_xz[:,z_Si,time],color="red")
    plt.plot(x,Jx_xz[:,z_Gi,time],color="blue")
    plt.plot(x,Jx_xz[:,z_Go,time],color="green")
    # plt.ylim([-0.75,0.75])
    # plt.xlim([x[0],x[-1]])
    plt.tight_layout()
ani = animation.FuncAnimation(fig, update,frames=int(nFrame/rate))
ani.save("Movie_IX_X_"+CirName+".gif", writer="imagemagick")


plt.show()
os.chdir("../plotProgram/")