import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation #アニメーション作成のためメソッドをインポート


from tqdm import tqdm
import time

from MakeSignal import *
from calc_board import calc_board
from calc_stick import calc_stick

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　パラメータの入力
#---------------------------------------------------------------------------------------------------------------------------------------------------
dx = 0.01 # x方向空間差分間隔[m]
dy = 0.01 # y方向空間差分間隔[m]
dz = 0.01 # z方向空間差分間隔[m]

c = 2.99792458e8 # 光速[m/s]
f = 1.0e9 # 周波数[Hz]

t = 0
dt = 0.99/(c * np.sqrt((1.0/dx ** 2 + 1.0/dy ** 2 + 1.0/dz ** 2))) #時間差分間隔[s]　#19.0657487[ps]

nx = 51 # x方向計算点数
ny = 51 # y方向計算点数
nz = 51 # z方向計算点数
nt = 150 # 計算ステップ数

#金属棒の設定(z軸)
in_stick = 11   #金属棒の始点
len_stick = 30  #金属棒の長さ[cm]
dis_stick = 0   #金属棒の間隔[cm]

in_board = 11   #金属板の始点
len_board = 30  #金属板の長さ[cm]
wid_board = 3   #金造版の幅[cm]
dis_board = 0   #金属板の間隔[cm]

in_current = 10  #入力する電流の位置


# 電気定数
eps=np.full((nx,ny,nz),8.854187817e-12)
mu=np.full((nx,ny,nz),1.2566370614e-6)
sigma=np.full((nx,ny,nz),0.0)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　meshの設定
#---------------------------------------------------------------------------------------------------------------------------------------------------
X1 = np.linspace(0, nx, nx+1)
Y1 = np.linspace(0, ny, ny+1)
Z1 = np.linspace(0, nz, nz+1)
#Y2, Z2 = np.meshgrid(Y1[:-1],Z1)
#X2, Y2 = np.meshgrid(X1,Y1)
X2, Z2 = np.meshgrid(X1,Z1)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　入力波形の設定
#---------------------------------------------------------------------------------------------------------------------------------------------------
#Signal = SquareWavePulse(1.0,dt*10,dt*10,dt*10,dt*20,dt,nt)
Signal = GaussianPulse(1.0,10*dt,40*dt,dt,nt)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#　電磁界計算
#---------------------------------------------------------------------------------------------------------------------------------------------------
#H_x, H_y, H_z, E_x, E_y, E_z, J_z = calc_stick(in_stick, len_stick, dis_stick, in_current, Signal, dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, sigma)
H_x, H_y, H_z, E_x, E_y, E_z, J_x, J_z = calc_board(in_board, len_board, wid_board, dis_board, in_current, Signal, dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, sigma)
    
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


fig = plt.figure(figsize=(12.5,10))    
nFrame = 300
rate = 3
def update(i):
    plt.cla()
    # ax = fig.gca(projection='3d')    
    time = rate*i
    print(time)
    ### Plot ###
    plt.imshow(J_x[:, (ny//2+1)+(dis_board//2), :, i],vmax=0.1,vmin=-0.1,cmap="bwr")
    # ax.set_zlim(zlim)
    plt.tight_layout()
ani = animation.FuncAnimation(fig, update,frames=int(nFrame/rate))
# ani.save("Movie.gif", writer="imagemagick")

plt.show()




