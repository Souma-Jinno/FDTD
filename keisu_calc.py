# from numba.np.ufunc import parallel
import numpy as np

from numba import jit,prange
from numpy.lib.function_base import append

# @jit(nopython=True,parallel=True)
def keisu_calc(dx, dy, dz, dt, eps, mu, rho):
   
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    #　係数の計算
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    dhx = dt / (mu * dx) # 式(9) - (11)の右辺係数
    dhy = dt / (mu * dy) # 式(9) - (11)の右辺係数
    dhz = dt / (mu * dz) # 式(9) - (11)の右辺係数
    ce = (2*eps - rho * dt) / (2*eps + rho * dt) # 式(12) - (14)の右辺第一項係数
    dex = 2*dt / ((2*eps + rho * dt) * dx) # 式(12) - (14)の右辺第二項係数
    dey = 2*dt / ((2*eps + rho * dt) * dy) # 式(12) - (14)の右辺第二項係数
    dez = 2*dt / ((2*eps + rho * dt) * dz) # 式(12) - (14)の右辺第二項係数
    de = 2*dt / (2*eps + rho * dt)
    dhy_Hx = append_mean(dhy,0)
    dhz_Hx = append_mean(dhy,0)
    dhx_Hy = append_mean(dhx,1)
    dhz_Hy = append_mean(dhz,1)
    dhy_Hz = append_mean(dhy,2)
    dhx_Hz = append_mean(dhx,2)
    return dhy_Hx,dhz_Hx,dhx_Hy,dhz_Hy,dhy_Hz,dhx_Hz,ce, dex, dey, dez, de


def append_mean(A,a):   # A : 3d-Array, a : axis(int)
    if a == 0: # x-axis
        temp = np.concatenate((A[0:1,:,:],A),axis=0)
        temp = np.concatenate((temp,A[-2:-1,:,:]),axis=0)
        B = (1/2)*(temp[1:,:,:] + temp[:-1,:,:])
    elif a == 1: # y-axis
        temp = np.concatenate((A[:,0:1,:],A),axis=1)
        temp = np.concatenate((temp,A[:,-2:-1,:]),axis=1)
        B = (1/2)*(temp[:,1:,:] + temp[:,:-1,:])
    elif a == 2: # z-axis
        temp = np.concatenate((A[:,:,0:1],A),axis=2)
        temp = np.concatenate((temp,A[:,:,-2:-1]),axis=2)
        B = (1/2)*(temp[:,:,1:] + temp[:,:,:-1])
    return B
