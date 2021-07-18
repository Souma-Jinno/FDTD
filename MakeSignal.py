import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation #アニメーション作成のためメソッドをインポート


from tqdm import tqdm
import time
import math
from numba import jit 

def SquareWavePulse(V,ts,tr,tf,tm,Δt,Nt):
    Signal = np.zeros([Nt])
    for m in range(Nt):
        time = m * Δt
        if time <= ts:
            Signal[m] = 0.0
        elif ts < time <= ts + tr:
            Signal[m] = (V/tr)*(time-ts)
        elif ts + tr < time <= ts+tr+tm:
            Signal[m] = V
        elif ts+tr+tm < time <= ts+tr+tm+tf:
            Signal[m] = V - (V/tf)*(time-(ts+tr+tm))
        elif ts+tr+tm+tf < time:
            Signal[m] = 0.0
    return Signal


def GaussianPulse(V,fwhm,myu,Δt,Nt):
    Signal = np.zeros([Nt])
    sigma = fwhm/math.sqrt(2*math.log(2))
    for m in range(Nt):
        time = m * Δt
        Signal[m] =V * math.exp((-(time-myu)**2)/(2 * sigma**2))
    return Signal