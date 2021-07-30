2021/07/18  
作成者：神野崇馬  

# 変更点
・薄さ0の平面導体を任意の場所に配置できるようにする。  
・エクセルファイルから任意の形状の導体配置と誘電率分布を入力できるようにした。  
・入力信号と入力箇所もエクセルファイルから入力できるようにした。  
・メモリ削減のため、時間のストックはせずに必要な配列だけ残す 

# 入力ファイル
CirNameで分別  
1. CircuitInformation{CirName}.xlsx
計算領域と導体の位置の入力
2. EpsilonMuInformation{CirName}.xlsx
比誘電率の空間分布を入力
3. InputInformation{CirName}.xlsx
入力信号の波形を入力

# jitのエラー例
・Segmentation fault: 11  のエラーが出るがエラー箇所がわからない
@jit外してデバッグ → エラー箇所を表示させる
・np.arrangeの要素指定するときにfloat型だった
・配列がオーバーフローしていた
・np.zeros()の配列の大きさはタプルでやる。np.zeros([x,y,z])→np.zeros((x,y,z))

# その他
・磁場のforループの数があっていない（適当にやってる）
↑ dhx,dhy,dhzの数
・空間の定義
[x,y,z] or [z,y,x]
[x,y,z]だと、plotする時に軸が逆になる。
・ipython内でrunしたあと、もう一度runすると、ERROR:root:File `'3D_FDTD_main.py'` not found.と出る
・pythonのバージョンでpandasの使い方が違う。エクセルのsheetの名前取り込む。.book.sheet_names()→.book.sheetnames