import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation #アニメーション作成のためメソッドをインポート

from MakeSignal import *
# from calc_board import calc_board
from calc_stick import calc_stick

import os
import pandas as pd

from make_input import MakeInput        # 入力配列の作成
from make_signal_matrix import MakeWaveFormMatrix
os.chdir("./circuitInformation/")

#---------------------------------------------------------------------------------------------------------------------------------------------------
# Load Circuit Information
#---------------------------------------------------------------------------------------------------------------------------------------------------
CirName = "1-1-1-1-1"
dx, dy, dz, dt, nx, ny, nz, nt, ε, μ, ρ, PIX,PIY = MakeInput(CirName)


#---------------------------------------------------------------------------------------------------------------------------------------------------
#　入力波形の設定
#---------------------------------------------------------------------------------------------------------------------------------------------------
InputInfo = pd.ExcelFile("InputInformation"+CirName+".xlsx")
Signal = MakeWaveFormMatrix(InputInfo,dt,nt)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　電磁界計算
#---------------------------------------------------------------------------------------------------------------------------------------------------
#H_x, H_y, H_z, E_x, E_y, E_z, J_z = calc_stick(in_stick, len_stick, dis_stick, in_current, Signal, dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, sigma)
H_x, H_y, H_z, E_x, E_y, E_z, J_x, J_z = calc_stick(Signal, Signal.shape[0],dx, dy, dz, dt, nx, ny, nz, nt, ε, μ, ρ, PIX, PIY)
    
#---------------------------------------------------------------------------------------------------------------------------------------------------
#　可視化
#---------------------------------------------------------------------------------------------------------------------------------------------------
# fig = plt.figure() #新規ウィンドウを描画
# fig.set_dpi(100)  #描画解像度の指定
# #ax = Axes3D(fig)  #3D軸の形成

# ims = []

# for i in range(nt-1):

#     if dis_board % 2 != 0:
#         dis_board = dis_board + 1

#     #img = plt.plot(J_z[nx//2+1,(ny//2+1)+(dis_board//2), :, i], color = 'g')
#     #img = plt.plot(J_x[:,(ny//2+1)+(dis_board//2), nz//2+1, i], color = 'g')
#     #img = plt.plot(H_y[:, ny//2+1, nz//2+1, i], color='g')
#     #img = plt.plot(H_x[nx//2+1, :, nz//2+1, i], color= 'g')
#     #img = plt.plot(E_z[:,ny//2+1, nz//2+1, i], color = 'g')

#     img = plt.imshow(J_x[:, (ny//2+1)+(dis_board//2), :, i], cmap='bwr')

#     #plt.title("YZplane")
#     #plt.title("XZplane")
#     #plt.title("XYplane")
#     plt.xlabel('x[cm]')
#     #plt.xlabel('y[cm]')
#     #plt.xlabel('z[cm]')
#     plt.ylabel('Jx[A/m^2]')
#     #plt.ylabel('Jz[A/m^2]')
#     #plt.ylabel('Hx[A/m]')  
#     #plt.ylim(-1.0,1.0)

#     ims.append(img)

# ani = animation.FuncAnimation(fig,ims)
# ani.save('anim.gif', writer="imagemagick")


# fig = plt.figure(figsize=(12.5,10))    
# nFrame = 300
# rate = 3
# def update(i):
#     plt.cla()
#     # ax = fig.gca(projection='3d')    
#     time = rate*i
#     print(time)
#     ### Plot ###
#     plt.imshow(J_x[:, (ny//2+1)+(dis_board//2), :, i],vmax=0.1,vmin=-0.1,cmap="bwr")
#     # ax.set_zlim(zlim)
#     plt.tight_layout()
# ani = animation.FuncAnimation(fig, update,frames=int(nFrame/rate))
# # ani.save("Movie.gif", writer="imagemagick")

# plt.show()




