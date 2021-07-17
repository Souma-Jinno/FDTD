# from numba.np.ufunc import parallel
import numpy as np

from numba import jit,prange

@jit(nopython=True,parallel=True)
def keisu_calc(dx, dy, dz, dt, eps, mu, sigma):
   
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    #　係数の計算
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    dhx = dt / (mu * dx) # 式(9) - (11)の右辺係数
    dhy = dt / (mu * dy) # 式(9) - (11)の右辺係数
    dhz = dt / (mu * dz) # 式(9) - (11)の右辺係数
    ce = (2*eps - sigma * dt) / (2*eps + sigma * dt) # 式(12) - (14)の右辺第一項係数
    dex = 2*dt / ((2*eps + sigma * dt) * dx) # 式(12) - (14)の右辺第二項係数
    dey = 2*dt / ((2*eps + sigma * dt) * dy) # 式(12) - (14)の右辺第二項係数
    dez = 2*dt / ((2*eps + sigma * dt) * dz) # 式(12) - (14)の右辺第二項係数
    de = 2*dt / (2*eps + sigma * dt)

    return dhx, dhy, dhz, ce, dex, dey, dez, de
