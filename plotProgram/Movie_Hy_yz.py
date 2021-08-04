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
Hy_yz    = Data1["Hy_yz"]
dt1      = Data1["dt"]


plt.rcParams['font.family'] ='Helvetica'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 18 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

fig=plt.figure(figsize=(7,5))




nFrame = 300
rate = 3
vmax=Hy_yz[30,:,:].max()/3
vmin=-Hy_yz[30,:,:].max()/3

def update(i):
    plt.clf()
    # time = rate*i+14000
    time = rate*i
    # t = dt*time
    print(time)
    ### Plot ###
    plt.imshow(Hy_yz[:,:,time].T,vmax=vmax,vmin=vmin,cmap="bwr",origin="lower")
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tight_layout()
ani = animation.FuncAnimation(fig, update,frames=int(nFrame/rate))
ani.save("Movie_Hy_yz_"+CirName+".gif", writer="imagemagick")


fig = plt.figure(figsize=(8,6))    
plt.imshow(Hy_yz[:,:,100].T,vmax=vmax,vmin=vmin,cmap="bwr",origin="lower")
plt.colorbar()
plt.savefig("Movie_Hy_yz_"+CirName+"_colorbar.png")
# plt.figure()
# plt.plot(IX_So+IX_Gi)

plt.show()
os.chdir("../plotProgram/")
