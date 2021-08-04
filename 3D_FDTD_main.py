import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation #アニメーション作成のためメソッドをインポート

from MakeSignal import *
from calc_arbitral import calc_arbitral
from keisu_calc import keisu_calc

import os
import pandas as pd

from make_input import MakeInput        # 入力配列の作成
from make_signal_matrix import MakeWaveFormMatrix
os.chdir("./circuitInformation/")

#---------------------------------------------------------------------------------------------------------------------------------------------------
# Load Circuit Information
#---------------------------------------------------------------------------------------------------------------------------------------------------
CirName = "1-2-1-1-1"
step_save = 10
dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, rho, PIX,PIY,PIZ = MakeInput(CirName)


#--------------------------------------------------------------------------------------------------------------------------------------------------
#　入力波形の設定
#---------------------------------------------------------------------------------------------------------------------------------------------------
InputInfo = pd.ExcelFile("InputInformation"+CirName+".xlsx")
Signal = MakeWaveFormMatrix(InputInfo,dt,nt)
nInput = Signal.shape[0]
#---------------------------------------------------------------------------------------------------------------------------------------------------
#　係数の計算
#---------------------------------------------------------------------------------------------------------------------------------------------------
dhy_Hx,dhz_Hx,dhx_Hy,dhz_Hy,dhy_Hz,dhx_Hz,ce,dex,dey,dez,de= keisu_calc(dx, dy, dz, dt, eps, mu, rho)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　電磁界計算
#---------------------------------------------------------------------------------------------------------------------------------------------------
Jx_yz,Hx_yz,Hy_yz,Hz_yz,Ex_yz,Ey_yz,Ez_yz,Jx_xz,Hx_xz,Hy_xz,Hz_xz,Ex_xz,Ey_xz,Ez_xz=calc_arbitral(CirName,step_save,Signal,nInput,dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, rho, PIX, PIY,PIZ,dhy_Hx,dhz_Hx,dhx_Hy,dhz_Hy,dhy_Hz,dhx_Hz,ce,dex,dey,dez,de)

np.savez_compressed(
            "3dFDTD_Circuit"+CirName+".npz",
            dx=dx,dy=dy,dz=dz,dt=dt,step_save=step_save,
            Jx_yz=Jx_yz,Hx_yz=Hx_yz,Hy_yz=Hy_yz,Hz_yz=Hz_yz,Ex_yz=Ex_yz,Ey_yz=Ey_yz,Ez_yz=Ez_yz,
            Jx_xz=Jx_xz,Hx_xz=Hx_xz,Hy_xz=Hy_xz,Hz_xz=Hz_xz,Ex_xz=Ex_xz,Ey_xz=Ey_xz,Ez_xz=Ez_xz
        )
#---------------------------------------------------------------------------------------------------------------------------------------------------
#　可視化
#---------------------------------------------------------------------------------------------------------------------------------------------------

# data = data3
# fig = plt.figure()    
# nFrame = 2600
# rate = 26
# vmax = data.max()/3
# vmin = -data.max()/3
# # vmax = 1e-5
# # vmin = -1e-5
# def update(i):
#     plt.cla()
#     # ax = fig.gca(projection='3d')    
#     time = rate*i
#     print(time)
#     ### Plot ###
#     # plt.imshow(data[:, : ,9, i],vmax=vmax,vmin=vmin,cmap="bwr")
#     # ax.set_zlim(zlim)
#     plt.imshow(data[:,:,i].T,vmax=vmax,vmin=vmin,cmap="bwr")
#     # plt.plot(data[:,14,9,i].T)
#     # plt.ylim([vmin,vmax])
#     plt.tight_layout()
# ani = animation.FuncAnimation(fig, update,frames=int(nFrame/rate))
# # ani.save("Movie.gif", writer="imagemagick")

# plt.show()




