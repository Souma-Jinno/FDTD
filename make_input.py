import pandas as pd
import numpy as np
from make_position_matrix import MakeIXPosition, MakeIYPosition



def MakeInput(CirName):
    # 電気定数
    ε0 = 8.854187817e-12
    μ0 = 1.2566370614e-6
    c0 = 1/np.sqrt(ε0*μ0)

    # 計算領域と導体配置のインポート
    CirInfo = pd.ExcelFile("CircuitInformation"+CirName+".xlsx")
    CirVal  = CirInfo.parse("Circuit Values",index_col=0)
    TotalTime  = float(CirVal.loc["Total Time"])   # 計算時間
    Width   = float(CirVal.loc["Width"])
    Length  = float(CirVal.loc["Length"])
    Height  = float(CirVal.loc["Height"])
    rho     = float(CirVal.loc["Rho"])
    temp    = np.array(CirInfo.parse("Layer1",index_col=0))
    nz      = int(CirVal.loc["Layer Number"])
    ny      = temp.shape[0]
    nx      = temp.shape[1]
    dx      = Width/nx
    dy      = Length/ny
    dz      = Height/nz

    # 計算領域内の導体の位置
    PC      = np.zeros([nx,ny,nz])              # Conductor Position
    LayerList = CirInfo.book.sheet_names()
    for i,sheet in enumerate(LayerList[1:]):    # LayerList[0] は Circuit Values
        PC[:,:,i] = np.array(CirInfo.parse(sheet,index_col=0))
    # Ex配列の導体の位置
    PIX     = MakeIXPosition(PC)
    PIY     = MakeIYPosition(PC)
    ρ = rho * PC

    
    # 計算領域ないの比誘電率と比透磁率の分布
    EpsInfo = pd.ExcelFile("EpsilonInformation"+CirName+".xlsx")
    εr = np.array(EpsInfo.parse("Epsilon Values",index_col=0).loc["Relative Epsilon"])
    μr = np.array(EpsInfo.parse("Epsilon Values",index_col=0).loc["Relative Mu"])
    ε  = np.zeros([nx,ny,nz])
    μ  = np.zeros([nx,ny,nz])
    for i,sheet in enumerate(LayerList[1:]):
        df_temp = EpsInfo.parse(sheet,index_col=0)
        for j,eps_r in enumerate (εr):
            df_temp = df_temp.replace(j,ε0*eps_r)   # df_tempの中身はint型
        ε[:,:,i] = np.array(df_temp)
    for i,sheet in enumerate(LayerList[1:]):
        df_temp = EpsInfo.parse(sheet,index_col=0)
        for j,mu_r in enumerate (μr):
            df_temp = df_temp.replace(j,μ0*mu_r)   # df_tempの中身はint型
        μ[:,:,i] = np.array(df_temp)

    c = c0/εr
    dt = 0.99/(c.max() * np.sqrt((1.0/dx ** 2 + 1.0/dy ** 2 + 1.0/dz ** 2))) #時間差分間隔[s]　
    nt = int(TotalTime/dt)

    return dx,dy,dz,dt,nx,ny,nz,nt,ε,μ,ρ,PIX,PIY