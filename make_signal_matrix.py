import pandas as pd
import numpy as np
import math
import sys


def MakeWaveFormMatrix(InputInfo,dt,nt):
    SignalInfo = InputInfo.parse("Input Signal",index_col=0)
    InputPosition   = InputInfo.parse("Input Position",index_col=0)
    nSource = len(SignalInfo)
    Signal = np.zeros([nSource,3+nt])   # 3はx,y,z座標
    if len(SignalInfo) == len(InputPosition):
        pass
    else:
        print("ERROR : The number of signal information you input does not match with input positon.")
        sys.exit()
    ### Make Wave Form Matrix ###
    for i in range(nSource):
        print("===== Wave Form Information" + str(i) + " =====")
        print(SignalInfo.loc[i,:])
        Signal[i,0] = InputPosition.loc[i,"x"]
        Signal[i,1] = InputPosition.loc[i,"y"]
        Signal[i,2] = InputPosition.loc[i,"z"]
        Signal[i,3:] = WaveForm(SignalInfo.loc[i,:],dt,nt)
    return Signal
    
def WaveForm(SignalInfo,dt,nt):
    if SignalInfo["Wave Shape"] == "Square Wave":
        Amp = SignalInfo["Amplitude"]
        WaitInt = int(SignalInfo["Wait Time"]/dt)
        RiseInt = int(SignalInfo["Rise Time"]/dt)
        FlatInt = int(SignalInfo["Flat Time"]/dt)
        FallInt = int(SignalInfo["Fall Time"]/dt)
        PeriodInt   = WaitInt + RiseInt + FlatInt + FallInt + FlatInt
        if PeriodInt > nt:
            PeriodInt = nt
        SW = np.zeros([PeriodInt])
        SG = np.zeros([nt])

        for i in range (PeriodInt):
            if i==0:
                SW[i] = 0.0
            elif i<WaitInt:
                SW[i] = 0.0
            elif i>=WaitInt and i<(WaitInt + RiseInt):
                SW[i] = Amp*((np.float(i)-WaitInt)/RiseInt)
            elif i>=(WaitInt + RiseInt) and i<(WaitInt + RiseInt + FlatInt):
                SW[i] = Amp
            elif i>=(WaitInt + RiseInt + FlatInt) and i < (WaitInt + RiseInt + FlatInt + FallInt):
                SW[i] = Amp*(1-(np.float(i)-(WaitInt + RiseInt + FlatInt))/FallInt)
            elif i >= (WaitInt + RiseInt + FlatInt + FallInt) and i < (WaitInt + RiseInt + FlatInt + FallInt + FlatInt):
                SW[i] = 0.0
            else:
                SW[i]=0.0

        if SignalInfo["Continuous"] == 1:
            nContinuous = int(nt/PeriodInt)
            for i in range(nContinuous):
                SG[i*PeriodInt:(i+1)*PeriodInt] = SW
            remain = nt - nContinuous*PeriodInt
            SG[PeriodInt*nContinuous:] = SW[:remain]
        if SignalInfo["Continuous"] == 0:
            SG[:PeriodInt] = SW
        return SG
    
    if SignalInfo["Wave Shape"] == "Gaussian Pluse":
        Amp = SignalInfo["Amplitude"]
        FWHM       = SignalInfo["Gaussian Pulse FWHM Time"]
        centerTime  = SignalInfo["Gaussian Pulse Maximum Time"]
        sigma = FWHM / (2 * math.sqrt(2*math.log(2)))
        GaussianPulseMat = np.zeros([nt])
        nPeriod = int(centerTime/dt * 2)
        SingleGaussianPulseMat = np.array(
                                    [
                                        Amp*math.exp(-(t*dt-centerTime)**2/(2*sigma**2))   \
                                        for t in range(nPeriod)
                                    ]
                                    )
        if SignalInfo["Continuous"] == 0:
            GaussianPulseMat[:nPeriod] = SingleGaussianPulseMat
        if SignalInfo["Continuous"] == 1:
            nContinuous = int(nt/nPeriod)
            for i in range(nContinuous):
                GaussianPulseMat[nPeriod*i:nPeriod*(i+1)] = SingleGaussianPulseMat
        return GaussianPulseMat
    
    if SignalInfo["Wave Shape"] == "Sinusoidal":
        f = SignalInfo["Frequency"]
        Amp = SignalInfo["Amplitude"]
        PeriodTime = 1/f
        nPeriodTime = PeriodTime/dt
        SW = np.zeros([nt])
        for i in range(nt):
            SW[i] = Amp * math.sin(2*math.pi/nPeriodTime*i)    
        return SW

    else:
        print("ERROR : This signal does not exist in list")
        sys.exit()
        
        
if __name__ == '__main__':
    os.chdir("../circuitInformation")
    dt = 1e-11
    nt = 10000
    InputInfo = pd.ExcelFile('BoundaryInformation1-4.xlsx')
    SignalShape = MakeWaveFormMatrix(InputInfo,dt,nt)
    os.chdir("../numericalProgram")
    