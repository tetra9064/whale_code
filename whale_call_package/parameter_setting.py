#parameter_setting
import numpy as np

def para(parameter_number):
    if parameter_number == '0':
        # フーリエ変換の初期設定
        sr = 200 #サンプリング周波数
        n_fft = 256 # データの取得幅
        hop_length = int(n_fft*0.01) # 次の取得までの幅
        fmin = 5 #最小周波数
        fmax = 40 #最大周波数
        # バンドパスフィルタの初期設定
        fpass = np.array([10, 40]) #通過域端周波数
        fstop = np.array([5, 50]) #阻止域端周波数
        gpass = 3   #通過域端最大損失[dB]
        gstop = 40   #阻止域最小損失[dB]
        #CFARの初期設定
        num_train = 800 #平均参照区間
        num_guard = 400 #平均参照除外区間
        rate_fa = 2.5e-3 #検出倍率

    elif parameter_number == '1':

        # フーリエ変換の初期設定
        sr = 200 #サンプリング周波数
        n_fft = 256 # データの取得幅
        hop_length = int(n_fft*0.01) # 次の取得までの幅
        fmin = 5 #最小周波数
        fmax = 40 #最大周波数
        # バンドパスフィルタの初期設定
        fpass = np.array([15, 35]) #通過域端周波数
        fstop = np.array([5, 50]) #阻止域端周波数
        gpass = 3   #通過域端最大損失[dB]
        gstop = 40   #阻止域最小損失[dB]
        #CFARの初期設定
        num_train = 800 #平均参照区間
        num_guard = 400 #平均参照除外区間
        rate_fa = 2.5e-3 #検出倍率

    elif parameter_number == '2':

        # フーリエ変換の初期設定
        sr = 200 #サンプリング周波数
        n_fft = 256 # データの取得幅
        hop_length = int(n_fft*0.01) # 次の取得までの幅
        fmin = 5 #最小周波数
        fmax = 40 #最大周波数
        # バンドパスフィルタの初期設定
        fpass = np.array([15, 25]) #通過域端周波数
        fstop = np.array([5, 50]) #阻止域端周波数
        gpass = 3   #通過域端最大損失[dB]
        gstop = 40   #阻止域最小損失[dB]
        #CFARの初期設定
        num_train = 800 #平均参照区間
        num_guard = 400 #平均参照除外区間
        rate_fa = 2.5e-3 #検出倍率

    return sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa

def make_parameter_log(workdirectory, sr, n_fft, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa):
    #各パラメータの設定をテキストファイルで出力
    with open(f'{workdirectory}/parameter_setting.txt', 'w', encoding='utf-8') as f:
        f.write(f'sr={sr}\nn_fft={n_fft}\nfmin={fmin}\nfmax={fmax}\nfpass={fpass}\nfstop={fstop}\ngpass={gpass}\ngstop={gstop}\nnum_train={num_train}\nnum_guard={num_guard}\nrate_fa={rate_fa}\n')
    