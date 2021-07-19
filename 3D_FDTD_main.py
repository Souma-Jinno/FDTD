import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation #アニメーション作成のためメソッドをインポート

from MakeSignal import *
# from calc_board import calc_board
from calc_stick import calc_stick
from keisu_calc import keisu_calc

import os
import pandas as pd

from make_input import MakeInput        # 入力配列の作成
from make_signal_matrix import MakeWaveFormMatrix
os.chdir("./circuitInformation/")

#---------------------------------------------------------------------------------------------------------------------------------------------------
# Load Circuit Information
#---------------------------------------------------------------------------------------------------------------------------------------------------
CirName = "1-1-1-1-1"
dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, rho, PIX,PIY = MakeInput(CirName)


#---------------------------------------------------------------------------------------------------------------------------------------------------
#　入力波形の設定
#---------------------------------------------------------------------------------------------------------------------------------------------------
InputInfo = pd.ExcelFile("InputInformation"+CirName+".xlsx")
Signal = MakeWaveFormMatrix(InputInfo,dt,nt)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　係数の計算
#---------------------------------------------------------------------------------------------------------------------------------------------------
dhy_Hx,dhz_Hx,dhx_Hy,dhz_Hy,dhy_Hz,dhx_Hz,ce,dex,dey,dez,de= keisu_calc(dx, dy, dz, dt, eps, mu, rho)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　電磁界計算
#---------------------------------------------------------------------------------------------------------------------------------------------------
#H_x, H_y, H_z, E_x, E_y, E_z, J_z = calc_stick(in_stick, len_stick, dis_stick, in_current, Signal, dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, sigma)
data1,data2 = calc_stick(Signal, Signal.shape[0],dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, rho, PIX, PIY,dhy_Hx,dhz_Hx,dhx_Hy,dhz_Hy,dhy_Hz,dhx_Hz,ce,dex,dey,dez,de)
    
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


fig = plt.figure()    
nFrame = 500
rate = 5
vmax = data2.max()/2
vmin = -data2.max()/2
def update(i):
    plt.cla()
    # ax = fig.gca(projection='3d')    
    time = rate*i
    print(time)
    ### Plot ###
    # plt.imshow(data2[:, : ,7, i],vmax=vmax,vmin=vmin,cmap="bwr")
    # ax.set_zlim(zlim)
    plt.plot(data1[:,14,7,i])
    plt.tight_layout()
ani = animation.FuncAnimation(fig, update,frames=int(nFrame/rate))
# ani.save("Movie.gif", writer="imagemagick")

plt.show()




