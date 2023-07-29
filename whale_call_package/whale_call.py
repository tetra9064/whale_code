# coding : UTF8

#目次
'''
import系
環境設定
CFAR
call数を元にグラフ作成
スペクトログラム作成
クラスタリング(良い結果を得られなかったため実行する必要なし)
CNNでノイズやセンタリングを除去
中心周波数を求める
'''

#ファイル処理や計算用の基本的なライブラリ
from xml.dom.expatbuilder import ParseEscape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import copy
import math
import shutil
import datetime
import glob
import os
import random
import re
import itertools
import pickle
import time
from tqdm import tqdm
import gc #緊急用
#wavファイルを取り扱うライブラリ
import librosa
from librosa import display #librosa.displayで指定してもエラーが発生するため
#FFT等の処理を行うライブラリ
from scipy import signal
from scipy.io.wavfile import write as wavwrite
#クラスタリングを行うライブラリ
from skimage import io
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
#CNNを行うライブラリ
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import load_model


class SetField():
    #環境設定

    def __init__(self, workdirectory, datadirectory):
        self.workdirectory = workdirectory
        self.datadirectory = datadirectory

    def make_directory(self):
        #作業用ディレクトリの作成
        os.makedirs(f"{self.workdirectory}", exist_ok=True)

    def get_wavfilelist(self):
        #wav形式のファイルの名前一覧を取得
        filelist = sorted(glob.glob(f'{self.datadirectory}/*.wav'))
        return filelist

    def pick_wavfilelist(self, filelist, month):
        #任意のwavファイルの名前一覧を取得
        '''
        filelist : 1日単位のwavファイルのリスト
        month : 取得したい月をリスト形式で表したもの
        '''
        picked_filelist = []
        month_num = len(month)
        
        for i in range(month_num):
            for j in filelist:
                if month[i] in j:
                    picked_filelist.append(j)
            
        return picked_filelist

    def random_pick_wavfilelist(self, filelist, pick_num):
        #ランダムなwavファイルの名前一覧を取得
        '''
        pick_num : ランダムに取得するwavファイルの個数
        '''
        pick = random.sample(list([i for i in range(len(filelist))]), pick_num)
        return pick



    def del_wavfile(self, filelist, del_list):
        #不要なwaveファイルをfilelistから削除
        '''
        del_list : 削除するwavファイルをリスト形式で。
        '''
        for i in del_list:
            for j in filelist:
                if i in j:
                    idx = filelist.index(j)
                    del filelist[idx]
        
        return filelist

    def timescales(self, filelist):
        #スペクトログラム等の表示のための時間スケールを作成
        day_scale = []
        name = (re.search(r"\d{6}", filelist[0])).group()
        count_scale = []
        scale_flag = 2
        count = 0
        month = name[:4]
        
        for day in filelist:
            name = (re.search(r"\d{6}", day)).group()
            count += 1
            if name[:4] != month:
                month = name[:4]
                if scale_flag % 2 == 1:
                    scale_flag -= 1
                else:
                    scale_flag += 1
                    day_scale.append(name)
                    count_scale.append(count)
        day_scale.append((re.search(r"\d{6}", filelist[-1])).group())
        count_scale.append(len(filelist))

        return day_scale, count_scale
    

#----------------------------------------------------------------------------------------------------#

class Cfar(SetField):
    #CFAR

    def __init__(self, workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa):
        #パラメータセッティング
        super().__init__(workdirectory, datadirectory)
        self.sr = sr #サンプリング周波数
        self.n_fft = n_fft #データの取得幅
        self.hop_length = hop_length #データの移動幅
        self.fmin = fmin #最小周波数
        self.fmax = fmax #最大周波数
        self.fpass = fpass #通過域端周波数
        self.fstop = fstop #阻止域端周波数
        self.gpass = gpass #通過域端最大損失[dB]
        self.gstop = gstop #阻止域最小損失[dB]
        self.num_train = num_train #平均参照区間
        self.num_guard = num_guard #平均参照除外区間
        self.rate_fa = rate_fa #検出倍率

    def get_wav(self, file):
        #音声ファイルを取得する。y=音声、sr=サンプリング周波数
        '''
        file : 読み込むwavファイルのpath
        '''
        y, sr = librosa.audio.load(path=file, sr=self.sr)
        return y, sr

    def bandpass(self, x, freqpass, freqstop, gpass, gstop):
        #バンドパスフィルタを適用する
        '''
        x : bandpassfilterを適用する音声。get_wavで読み込んだもの。
        '''
        fn = self.sr / 2   #ナイキスト周波数
        wp = freqpass / fn  #通過域端周波数を正規化
        ws = freqstop / fn  #阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop) 
        b, a = signal.butter(N, Wn, "band")          
        y = signal.filtfilt(b, a, x)
                 
        return y



    def detect_peaks(self, x):
        #ピークを検出するCFAR
        num_cells = x.size
        num_train_half = round(self.num_train / 2)
        num_guard_half = round(self.num_guard / 2)
        num_side = num_train_half + num_guard_half

        alpha = self.num_train*(self.rate_fa**(-1/self.num_train) - 1) # threshold factor

        peak_idx = []

        for i in range(num_side, num_cells - num_side):
            #指定範囲内の最大値のインデックス番号を取得し、i-num_sideを足した数がiと同値なら省く。
            if i != i-num_side+np.argmax(x[i-num_side:i+num_side+1]): 
                continue

            sum1 = np.sum(x[i-num_side:i+num_side+1])
            sum2 = np.sum(x[i-num_guard_half:i+num_guard_half+1]) 
            p_noise = (sum1 - sum2) / self.num_train #p_noiseが周辺区間の平均ノイズ
            threshold = alpha * p_noise #閾値

            if x[i] > threshold: 
                peak_idx.append(i)

        return peak_idx

    def make_cfar_dir(self, i):
        #cfarしたものを格納するディレクトリを作成
        '''
        i : wavファイル名
        '''
        name = (re.search(r"\d{6}", i)).group()
        os.makedirs(f'{self.workdirectory}/cfar/{name}', exist_ok=True)
        os.makedirs(f'{self.workdirectory}/cfar/{name}/split', exist_ok=True)

        return name

    def run_cfar(self, data, name):
        #ピーク検出、結果出力を含むCFARの一連の処理を実行する
        '''
        data : ピーク検出するwavファイル。バンドパスフィルタをかけてから行う。
        name : yymmdd
        '''
        peak_idx = self.detect_peaks(np.abs(data)) #CFAR
        
        #1日毎のパラメータ、callの時間、何番目の要素かをテキストファイルで出力する
        with open(f'{self.workdirectory}/cfar/{name}/peak.txt', 'w',encoding='utf-8', newline='\n') as f:
            f.write(f'検出数：{str(len(peak_idx))} num_train={self.num_train} num_guard={self.num_guard} rate_fa={self.rate_fa}\nfmin={self.fmin} fmax={self.fmax} fpass={self.fpass} fstop={self.fstop} gpass={self.gpass} gstop={self.gstop}\n')
            for i in peak_idx:
                tm = datetime.timedelta(seconds=int(i/200))
                f.write(str(tm) + ' ' + str(i) +'\n')
        
        #処理を行ったファイル名をテキストファイルで出力する。追記なので注意。
        with open(f'{self.workdirectory}/cfar/pick_list.txt', 'a', encoding='utf=8') as f:
            print(f'{self.workdirectory}/cfar/{name}', file=f)

        return peak_idx



    def make_split_wavfile(self, time, peak_idx, name, data):
        #取得したピークを中心に任意の秒数のwavファイルをピークの数だけ作成する
        '''
        time : ひとつの鳴音のwavファイルの長さ。鳴音を中心にtime分のwavファイルを作成する。
        peak_idx : run_cfarで作成した1日間のpeakを記録したもの。
        name : yymmdd
        data : wavファイル。ここではスペクトログラム作成用のバンドパスフィルタを適用するため、生の音声データを入れる。
        '''
        #バンドパスフィルタを適用する
        y = self.bandpass(x=data, freqpass=np.array([10,40]), freqstop=np.array([5,50]), gpass=self.gpass, gstop=self.gstop)
                 
        #peak_idxを元に指定時間のwavファイルを作成する
        for i in range(len(peak_idx)):
            name_idx = str(i)
            while len(name_idx) < 6:
                name_idx = '0' + name_idx

            wavwrite(f'{self.workdirectory}/cfar/{name}/split/{name}_{name_idx}.wav',self.sr, y[int(peak_idx[i])-int((self.sr*time/2)):int(peak_idx[i])+int((self.sr*time/2))])



    def mark_calls_one_day(self, handle_day, filelist):
        #振幅のグラフにcfarで抽出したピークにマークする。単日検証用
        '''
        handle_day : 検証する日。yymmdd
        '''
        
        #wavファイルを取得
        target_file = [i for i in filelist if f"{handle_day}" in i]
        if len(target_file) > 1:
            print('目的のファイルが多数検出されています')
        y, sr = librosa.audio.load(path=target_file[0] ,sr=self.sr)

        #バンドパスフィルタを適用し、最大値で正規化
        data = self.bandpass(x=y, freqpass=self.fpass, freqstop=self.fstop, gpass=self.gpass, gstop=self.gstop)
        data = data/np.max(data)

        #ピークの場所を取得し、描画する準備
        with open(f'{self.workdirectory}/cfar/{handle_day}/peak.txt', 'r', encoding='utf-8') as f:
            peaks = []
            for line in f:
                line = line.rstrip()
                peaks.append(line)
        peaks = peaks[2:]
        peaks_sp = [i.split(' ') for i in peaks]
        peak_idx = [int(i[1]) for i in peaks_sp]
        ydy = np.max(y)
        ydd = np.max(data)
        peak_pw = [y[i]/ydy for i in peak_idx]
        peak_pw_d = [data[i]/ydd for i in peak_idx]
        y_x_range = np.array(range(0,len(y),1), dtype='uint32')

        #波形をplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, y, zorder=1)
        plt.show()
        #波形を絶対値でplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, np.abs(y), zorder=1)
        plt.show()
        #バンドパスフィルタを適用した波形をplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, data, zorder=1)
        plt.show()
        #バンドパスフィルタを適用した波形に絶対値でplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, np.abs(data), zorder=1)
        plt.show()
        #波形をplotしたものにpeakを入れたものをplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, y, zorder=1)
        plt.scatter(peak_idx, peak_pw, color="red", zorder=2)
        plt.show()
        #波形をplotしたものにpeakを入れたものを絶対値でplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, np.abs(y), zorder=1)
        plt.scatter(peak_idx, np.abs(peak_pw), color="red", zorder=2)
        plt.show()
        #バンドパスフィルタを適用した波形にpeakを入れたものをplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, data, zorder=1)
        plt.scatter(peak_idx, peak_pw_d, color="red", zorder=2)
        plt.show()
        #バンドパスフィルタを適用したものにpeakを入れたものを絶対値でplot
        plt.figure(figsize=[10,4])
        plt.plot(y_x_range, np.abs(data), zorder=1)
        plt.scatter(peak_idx, np.abs(peak_pw_d), color="red", zorder=2)
        plt.show()

    def make_call_graph(self, filelist, peak_file, dir_name, count_scale,day_scale):
            
            #分割数。1時間ごとならsp_time=24,sp_num=分割する時間の秒数*200。１時間なら3600*200
            split_time = 24
            split_num = 720000
            time = 'hour'

            #各日の最初の要素を１つ後の要素と同じにする(1でTrue)
            first_split = 0

            makecallgraph = MakeCallGraph(workdirectory=self.workdirectory, datadirectory=self.datadirectory, peak_file=peak_file, dir_name=dir_name)
            max_day_call = makecallgraph.get_max_call_num(filelist)
            months, mean_list, month_name_list = makecallgraph.make_months_list(filelist=filelist)

            max_per_hour = []
            peak_num_per_day = []
            max_mean = []
            max_mean_per = []
            month_peaks = []
            for i in range(len(months)):
                peakn_hour, mean, mean_per, month_peak_ave, mean_list, peak_num_for_bar = makecallgraph.cal_graph_data(write_setting=i, filelist=filelist, days=months[i][1], month=months[i][0], split_time=split_time, split_num=split_num, time=time, first_split=first_split, mean_list=mean_list)
                max_per_hour.append(peakn_hour)
                max_mean_per.append(mean_per)
                month_peaks.append(month_peak_ave)
                peak_num_per_day.append(peak_num_for_bar)
                max_mean.append(mean)
            max_hour_call = max(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(max_per_hour)))))
            max_mean_per_call_list = []
            max_mean_call_list = []
            for i in max_mean_per:
                max_mean_per_call_list.append(max(i))
            max_mean_per_call = max(max_mean_per_call_list)

            for i in max_mean:
                max_mean_call_list.append(max(i))
            max_mean_call = max(max_mean_call_list)

            mean_list = [[[] for j in range(24)] for i in range(len(months))]

            for i in range(len(months)):
                makecallgraph.detect_call_make_glaph(write_setting=i,filelist=filelist, days=months[i][1], month=months[i][0], split_time=split_time, split_num=split_num, time=time, first_split=first_split, mean_list=mean_list,max_hour_call=max_hour_call,max_mean_call=max_mean_call,max_mean_per_call=max_mean_per_call)

            makecallgraph.make_month_call_num(months=months, month_peak_ave=month_peaks, first_split=first_split, time=time, filelist=filelist, peak_num_per_day=peak_num_per_day, max_day_call=max_day_call,count_scale=count_scale,day_scale=day_scale)

            makecallgraph.call_num_per_hour(mean_list=mean_list, month_name_list=month_name_list, time=time, first_split=first_split, max_hour_call=max_hour_call)

#----------------------------------------------------------------------------------------------------#

class MakeCallGraph(SetField):
    #Call数のグラフ作成

    def __init__(self, workdirectory, datadirectory, peak_file, dir_name):
        super().__init__(workdirectory, datadirectory)

        self.peak_file = peak_file
        self.dir_name = dir_name

        os.makedirs(f'{self.workdirectory}/{self.dir_name}/', exist_ok=True)
        with open(f'{self.workdirectory}/{self.dir_name}/data_explanation.txt', 'w', encoding='utf=8') as f:
            f.write('先頭にf0とあるディレクトリはセンタリングの影響を考え、0時から1時のデータを1時から2時のデータに変換したもの。\n\n\
月名のディレクトリは各時間の月平均、それを割合にしたオレンジの図、各日のグラフで構成されている。\n\n\
ALLhourは月平均、その割合をまとめたもの。peak_num.txtは月毎の1日平均数\n\n\
month_mean_per_hourは月毎に同時刻の変化を追ったもの\n\n\
day_peaks.jpgは単純に全ての日の数を表示、month_peak_ave.jpgは月毎の1日の平均数。日数が少ない月でも1日の平均数を比較可能\n\n\
month_sum.jpgは月毎の総数')





    def get_max_call_num(self,filelist):
        #１日のcall数の最大値を取得
        month_call_list=[]
        for i in filelist:
            name = (re.search(r"\d{6}", i)).group()
            with open(f'{self.workdirectory}/cfar/{name}/{self.peak_file}', 'r',encoding='utf-8', newline='\n') as f:
                month_call_list.append(f.readline())
        
        for i in range(len(month_call_list)):
            month_call_list[i] = month_call_list[i].split(' ')
            month_call_list[i] = month_call_list[i][0]
            month_call_list[i] = int(re.search(r'\d+', month_call_list[i]).group())

        max_day_call = max(month_call_list)
        max_day_call_idx = month_call_list.index(max_day_call)
        print(filelist[max_day_call_idx], max_day_call)

        return max_day_call            


    
    def make_months_list(self, filelist):
        #実行する月のリストを作成し、月ごとの日数を取得する
        days = []
        months = []
        for i in filelist:
            yymmdd = re.search(r"\d{6}", i).group()
            days.append(yymmdd)

        count = 0    
            
        while len(days) != 0:
            yymm = re.search(r"\d{4}", days[0]).group()
            months.append([yymm, 0])
            
            for i in days:
                if yymm in i:
                    months[count][1] += 1
            for i in range(months[count][1]):        
                days.pop(0)
            
            count += 1
        

        mean_list = [[[] for j in range(24)] for i in range(len(months))]

        month_name_list=[]
        for i in range(len(months)):
            month_name_list.append(months[i][0])
        
        #months : [[yymm, n_day], [yymm, n_day]...]
        #mean_list : [[]*24]*月数

        return months, mean_list, month_name_list



    def cal_graph_data(self, write_setting,filelist, days, month, split_time, split_num, time, first_split, mean_list):
        #ピークを元にグラフを作成する
        '''
        write_setting : 何個目の月か。make_month_listで作成したmonthのrange(len())
        days : month[i][1] 該当月の日数。
        month : month[i][0] yymm
        split_time : 何分割にするか。デフォルトで1日を24分割
        split_num : split_timeでの分割1つ分の秒数*サンプリング周波数(1分割あたりのサンプル数)
        time : 1時間で分割しているためhourという名前
        first_split : 0or1, centeringでグラフが分かりにくい時、0-1時間を1-2時間と同じにする。基本オフ(0)。
        mean_list : [[]*24]*月数, make_month_listで作成したもの
        '''

        if first_split == 1:
            time = 'f0'+time
        
        #ファイル名を年月日の数字６文字に変換する。
        picked = []
        for i in filelist:
            if '\\' + month in i: #うまくファイル名のyymmdd-のyymmだけ引っかかるように'\\'を'/'とかに調整する。基本filelist内の絶対パスのファイル名のyymmddの一つ手前を指定するとよい

                m = re.search(r"\d{6}", i)
                pick = m.group()
                picked.append(pick)             

        os.makedirs(f'{self.workdirectory}/{self.dir_name}/{time}', exist_ok=True)
        peak_hour = [[[] for j in range(split_time)] for i in range(days)] #[[[]*24]*月の日数]
        peakn_hour = [[] for j in range(days)] #[[]*月の日数]
        month_peak_ave = 0
        peak_num_for_bar = []

        for day in range(len(picked)):

            #peakの時間を記録したテキストファイルを開く
            with open(f'{self.workdirectory}/cfar/{picked[day]}/{self.peak_file}', 'r', encoding='utf-8') as f:
                peaks = []
                for line in f:
                    line = line.rstrip()
                    peaks.append(line)

            #cfarで出力したテキストファイルをコピーし、各種グラフを作成していく
            os.makedirs(f'{self.workdirectory}/{self.dir_name}/{time}/{month}{time}', exist_ok = True)
            shutil.copy(f'{self.workdirectory}/cfar/{picked[day]}/{self.peak_file}', f'{self.workdirectory}/{self.dir_name}/{time}/{month}{time}/peak_{picked[day]}.txt')
            
            peak_num_for_bar.append(peaks.pop(0))
            peaks.pop(0)
            month_peak_ave =+ len(peaks)                      

            for i in range(len(peaks)):
                peaks[i] = peaks[i].split(' ')
            
            #peaks_n : peak.txtの右側。
            peaks_n = []

            for i in peaks:
                peaks_n.append(i[1])
            
            for i in peaks_n:
                i = int(i)
                flag = 0
                #split_num : 1h*sr
                if i // split_num == flag:
                    pass
                else:
                    flag = i // split_num
                if 0 + flag*split_num <= i <=split_num+flag*split_num:
                    peak_hour[day][flag].append(i)
                #peak_hour : [[[peak, peak, peak...]*24]*月の日数]

            for i in peak_hour[day]:
                peakn_hour[day].append(len(i))
                #peakn_hour : [[len(call数)*24]*月の日数]

            if first_split == 1:
                peakn_hour[day][0] = peakn_hour[day][1]
        
        
        #1日毎の総数をリスト化
        for i in range(len(peak_num_for_bar)):
            peak_num_for_bar[i] = peak_num_for_bar[i].split(' ')
            peak_num_for_bar[i] = peak_num_for_bar[i][0]
            peak_num_for_bar[i] = int(re.search(r'\d+', peak_num_for_bar[i]).group())
        
        #月平均を一時間ごとに表示
        os.makedirs(f'{self.workdirectory}/{self.dir_name}/{time}/ALL{time}', exist_ok=True)

        if write_setting == 0:
            with open(f'{self.workdirectory}/{self.dir_name}/{time}/ALL{time}/peak_num.txt', 'w', encoding='utf-8') as f:
                f.write(month + ' ' + str(month_peak_ave) + '\n')
        else:
            with open(f'{self.workdirectory}/{self.dir_name}/{time}/ALL{time}/peak_num.txt', 'a', encoding='utf-8') as f:
                f.write(month + ' ' + str(month_peak_ave) + '\n')        

        peak_ave = [[] for i in range(split_time)]

        for i in range(len(peakn_hour)):
            for j in range(len(peakn_hour[i])):
                peak_ave[j].append(peakn_hour[i][j])

        mean = []
        for i in peak_ave:
            mean.append(statistics.mean(i))


        peak_per = [[] for i in range(split_time)]

        for i in range(len(peakn_hour)):
            for j in range(len(peakn_hour[i])):
                if peak_num_for_bar[i] == 0:
                    peak_per[j].append(0)
                else:
                    peak_per[j].append((peakn_hour[i][j]/peak_num_for_bar[i])*100)

        mean_per = []
        for i in peak_per:
            mean_per.append(statistics.mean(i))


        #１時間の変化を月ごとに見る
        for i in range(24):
            mean_list[write_setting][i].append(mean[i])    

        
        return peakn_hour, mean, mean_per, month_peak_ave, mean_list, peak_num_for_bar    
            


    def detect_call_make_glaph(self,write_setting,  filelist, days, month, split_time, split_num, time, first_split, mean_list,max_hour_call,max_mean_call, max_mean_per_call):
        #各種グラフの作成
        '''
        write_setting : 何個目の月か。make_month_listで作成したmonthのrange(len())
        days : month[i][1] 該当月の日数。
        month : month[i][0] yymm
        split_time : 何分割にするか。デフォルトで1日を24分割
        split_num : split_timeでの分割1つ分の秒数*サンプリング周波数(1分割あたりのサンプル数)
        time : 1時間で分割しているためhourという名前
        first_split : 0or1, centeringでグラフが分かりにくい時、0-1時間を1-2時間と同じにする。基本オフ(0)。
        mean_list : [[]*24]*月数, make_month_listで作成したもの
        max_hour_call : 分割した指定時間ごと１日の鳴音数
        max_mean_call : 分割した指定時間ごとの月の鳴音数(平均値)
        max_mean_per_call : 分割した指定時間ごと１日の鳴音数を時間ごとに%に変換。各時間の月の平均値
        '''
        
        peakn_hour, mean, mean_per, month_peak_ave, mean_list, peak_num_for_bar = self.cal_graph_data(write_setting, filelist, days, month, split_time, split_num, time, first_split, mean_list)

        #各日の指定時間ごとのcall数をグラフで表示
        for i in range(len(peakn_hour)):
            plt.figure()
            plt.plot(peakn_hour[i])
            plt.title('day' + str(i+1))
            plt.xlabel(time, size='xx-large')
            plt.ylabel('num', size='xx-large')
            plt.ylim(0, max_hour_call)
            plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/{month}{time}/day{str(i+1)}.jpg')
            plt.close()

        #月すべての同じ時間の平均を１時間ごとに表示
        plt.figure()
        plt.plot(mean)
        plt.title(f'{time}_ave')
        plt.xlabel(time, size='xx-large')
        plt.ylabel('num', size='xx-large')
        plt.ylim(0, max_mean_call)
        plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/{month}{time}/{month}month_ave.jpg')
        plt.close()
        #callの多い日の影響が大きく、少ない日の影響が小さい

        #各日で%に変換し、１時間ごとに平均を表示
        plt.figure()
        plt.plot(mean_per,color='orange')
        plt.title(f'{time}_per_ave')
        plt.xlabel(time, size='xx-large')
        plt.ylabel('%', size='xx-large')
        plt.ylim(0, max_mean_per_call)
        plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/{month}{time}/{month}month_per_ave.jpg')
        plt.close()
        #callの多い少ないに関わらず、全ての日が同じ力を持って平均化される

        #上記２つのグラフを単一のグラフに表示
        fig = plt.figure(16.8)

        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        ax1.plot(mean, label='mean')
        ax2.plot(mean_per, color='orange', label='per')
        ax1.grid()
        ax1.set_title(f'{month}_ave/per_ave')
        ax1.set_xlabel(time, size='xx-large')
        ax1.set_ylabel('num', size='xx-large')
        ax1.set_ylim(0, max_mean_call)
        ax2.set_ylim(0, max_mean_per_call)
        ax2.set_ylabel('%', size='xx-large')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')
        plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/{month}{time}/{month}month_per_ave_ave.jpg')
        plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/ALL{time}/{month}month_per_ave_ave.jpg')
        plt.close()



    def make_month_call_num(self, months, month_peak_ave, first_split, time, filelist, peak_num_per_day,max_day_call,count_scale,day_scale):
        #月ごとの総数、観測期間の総数を表示する
        '''
        month_peak_ave : 月ごとの１日辺りの鳴音数の平均値
        peak_num_per_day : 1日毎の鳴音数
        max_day_call : 全期間の１日ごとの鳴音数
        count_scale,day_scale : SetFieldクラスで作成したもの
        '''

        month_list = [i[0] for i in months]
        
        plt.figure()
        plt.bar(month_list,month_peak_ave)
        plt.title('month_call_ave')
        plt.xlabel('month', size='xx-large')
        plt.ylabel('num', size='xx-large')
        if first_split == 1:
            time = 'f0'+time
        plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/month_peak_ave.jpg')
        plt.close()

        month_sum=[]
        for i in peak_num_per_day:
            month_sum.append(sum(i))

        plt.figure()
        plt.bar(month_list,month_sum)
        plt.title('month_sum')
        plt.xlabel('month', size='xx-large')
        plt.ylabel('num', size='xx-large')
        if first_split == 1:
            time = 'f0'+time
        plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/month_sum.jpg')
        plt.close()

        peak_num_per_day_flat = list(itertools.chain.from_iterable(peak_num_per_day))

        plt.figure()
        plt.bar(filelist, peak_num_per_day_flat)
        plt.title('day_call_num')
        plt.xlabel('day', size='xx-large')
        plt.ylabel('num', size='xx-large')
        plt.ylim(0, max_day_call)
        plt.xticks(count_scale,day_scale)
        if first_split == 1:
            time = 'f0'+time
        plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/day_peaks.jpg')
        plt.close()

        with open(f'{self.workdirectory}/{self.dir_name}/{time}/peaks_num.txt', 'w', encoding='utf-8') as f:
            f.write('総鳴音数:' + str(sum(month_sum))+ '\n')
            f.write('月:月毎の日平均：月の総数\n')
            for i,j,k in zip(month_list,month_peak_ave,month_sum):
                f.write(i+' : '+str(j)+' : ' + str(k) + '\n')        



    def call_num_per_hour(self, mean_list, month_name_list, time, first_split,max_hour_call):
        #１時間の変化を月ごとに見る
        '''
        mean_list : 月毎の指定時間毎の鳴音数の平均値
        month_name_list : 月の名前
        max_hour_call : 指定時間毎の鳴音数
        '''
        if first_split == 1:
            time = 'f0'+time

        os.makedirs(f'{self.workdirectory}/{self.dir_name}/{time}/month_mean_per_hour/', exist_ok=True)
        loc = list(range(0,len(mean_list)))

        for i in range(24):
            mean = []
            for j in range(len(mean_list)):
                mean.append(mean_list[j][i])
            plt.figure()
            plt.plot(mean)
            plt.title('month_call_per_hour')
            plt.xlabel(f'{i}-{i+1}', size='xx-large')
            plt.ylabel('num', size='xx-large')
            plt.xticks(loc,month_name_list)
            plt.ylim(0, max_hour_call)
            plt.savefig(f'{self.workdirectory}/{self.dir_name}/{time}/month_mean_per_hour/{i}-{i+1}.jpg')
            plt.close()

    
    
#----------------------------------------------------------------------------------------------------#


class MakeSpectrogram(Cfar):
    #スペクトログラムの作成

    def __init__(self, workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa):
        super().__init__(workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa)
    
    def get_sp_wav(self, filelist):
        with open(f'{self.workdirectory}/cfar/pick_list.txt', 'r', encoding='utf-8') as f:
            picked = []
            for line in f:
                picked.append(line.strip()+ '/split/')
        
        if len(filelist) != len(picked):
            print('filelistとpick_listの数が合っていません。pick_listを確認してください')
        
        return picked

    def make_sp_spec_all(self, day):
        #全てのファイルに対してスペクトログラム画像を作成する。
        '''
        day : f'{workdirectory}/cfar/yymmdd' pick_list.txtを読み込んでリストにしたものの1要素
        '''
        
        #画像サイズの指定
        figsize_px = np.array([100,100])
        dpi = 96
        figsize_inch = figsize_px/dpi

        plt.rcParams['figure.figsize'] = figsize_inch
        plt.rcParams['figure.dpi'] = dpi

        splitfilelist = sorted(glob.glob(f'{day}*.wav'))
        name = (re.search(r"\d{6}", day)).group()
        os.makedirs(f'{self.workdirectory}/cfar/{name}/fig', exist_ok=True)
        

        #音声ファイルの取得、STFT、複素数部分と分割、dBスケールに変換
        for i in tqdm(range(len(splitfilelist)),desc=f'{name}'):
            data, sr =librosa.audio.load(splitfilelist[i], sr=self.sr)
            c = np.abs(librosa.stft(data,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
            C, phase = librosa.magphase(c)
            Cdb = librosa.amplitude_to_db(np.abs(C), ref=np.max)
            
            #スペクトログラムを作成、保存する            
            librosa.display.specshow(data=Cdb, sr=self.sr,fmin=10, fmax=self.fmax, x_axis='time', y_axis='hz')
            plt.ylim(10, self.fmax)
            plt.axis('off')
            plt.savefig(f'{self.workdirectory}/cfar/all/fig_convert/{name}_{i}.jpg.jpg')
            plt.cla()
            
            

    def make_spec(self, data):
        #単日検証用
        '''
        data : スペクトログラムに変換するwavファイル。librosaで読み込んだもの.
        '''

        c = np.abs(librosa.stft(data,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
        C, phase = librosa.magphase(c)
        Cdb = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        plt.figure(figsize=[12, 12])
        librosa.display.specshow(data=Cdb, sr=10,fmin=self.fmin, fmax=self.fmax, x_axis='time', y_axis='hz')
        plt.ylim(self.fmin, self.fmax)
        plt.axis('off')
        plt.show()

    def make_sep_spec(self, data, separate_spec):
        #1つの音声ファイルを任意の区間に区切ってスペクトログラムを表示する
        '''
        separate_spec : int, wavファイルをseparate_spec数に分割してそれぞれのスペクトログラムを表示。例として24に設定し、1日分のwavファイルで処理すると1時間毎のスペクトログラムを得られる。
        '''

        if separate_spec == 0:
            self.make_spec(data)
        else:
            for i in range(separate_spec):
                sep_data = data[int(i*len(data)/separate_spec):int((i+1)*len(data)/separate_spec)+1]
                self.make_spec(sep_data)



            
            
        
#----------------------------------------------------------------------------------------------------#


class Cnn(Cfar):
    #CNNを行う
    def __init__(self, workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa):
        super().__init__(workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa)

    def prepare_cnn(self, sr, dir_list):
        #CNNを行うために準備をする
        os.makedirs(f'{self.workdirectory}/cfar/all/random', exist_ok=True)
        date_list = os.listdir(f'{self.workdirectory}/cfar')
        if 'all' in date_list:
            date_list.remove('all')

        if  'pick_list.txt' in date_list:
            date_list.remove('pick_list.txt')
        
        date_list = list(map(int, date_list))
        date_list.sort()
        date_list = list(map(str,date_list))

        for i in dir_list:
            os.makedirs(f'{self.workdirectory}/cfar/all/train_ivent/{i}', exist_ok=True)     

        calls = []
        with open(f'{self.workdirectory}/cfar/all/call_list.txt', 'r', encoding='utf-8') as f:
            for line in f:
                calls.append(line)

        for date in date_list:
            peaks = []
            with open(f'{self.workdirectory}/cfar/{date}/peak.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    peaks.append(line)
                peaks = peaks[2:]
            peaks = [i.split(' ') for i in peaks]
            
      
            hour_list = []
            day_calls = int(len(peaks)/1000)
            if day_calls == 0:
                day_calls = 1
            for i in range(day_calls):
                pick_num = random.randint(0,len(peaks)-1)
                if int(peaks[pick_num][1]) in hour_list:
                    pass
                else:
                    if f'{date}_{pick_num}.jpg.jpg' in calls:
                        pass
                    else:
                        hour_list.append(int(peaks[pick_num][1]))
                        shutil.copyfile(f'{self.workdirectory}/cfar/all/fig_convert/{date}_{pick_num}.jpg.jpg', f'{self.workdirectory}/cfar/all/random/{date}_{pick_num}.jpg.jpg')

    def set_call_list(self):
        #既に抽出した画像をメモする。
        flag = False
        if 'call_list.txt' in os.listdir(f'{self.workdirectory}/cfar/all'):
            flag = True
        else:
            flag = False
        for i in os.listdir(f'{self.workdirectory}/cfar/all/train_ivent'):
            calls = os.listdir(f'{self.workdirectory}/cfar/all/train_ivent/{i}')
            call_list = []
            if flag:
                with open(f'{self.workdirectory}/cfar/all/call_list.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        call_list.append(line.rstrip('\n'))
            with open(f'{self.workdirectory}/cfar/all/call_list.txt', 'a', encoding='utf-8') as f:
                for call in calls:
                    if call in call_list:
                        pass
                    else:
                        f.write(call+'\n')


    def dir_set(self, dir_list):
        #CNN用にディレクトリをセット。
        train_dir = f'{self.workdirectory}/cfar/all/train_ivent' #訓練用データを保管
        val_dir = f'{self.workdirectory}/cfar/all/target'    #検証用データを保管
        backup_dir = f'{self.workdirectory}/cfar/all/model'  #作成したモデルを保管

        for i in dir_list:
            os.makedirs(f'{val_dir}/{i}', exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        
        #検証用データが存在しない場合、訓練用データから下記の割合で移動させる。
        val_data_split = 0.2
        if len(os.listdir(f'{val_dir}/{dir_list[0]}'))==0:
            train_list = os.listdir(train_dir)

            for group in train_list:
                group_list = os.listdir(f'{train_dir}/{group}')
                for i in range(int(len(group_list)*val_data_split)):
                    val = random.choice(group_list)
                    shutil.move(f'{train_dir}/{group}/{val}', f'{val_dir}/{group}/')
                    if val in group_list:
                        group_list.remove(val)
                
        return train_dir, val_dir, backup_dir

    
            

    def cnn_el_sentering(self, learning_rate, epochs, batch_size, image_size, train_dir, val_dir, backup_dir):
        #CNNモデルを作成する
        labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        labels.sort()

        num_classes = len(labels)

        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

        os.makedirs(backup_dir)

        with open(backup_dir + '/labels.txt','w') as f:
            for label in labels:
                f.write(label+"\n")
        
        train_data_gen = ImageDataGenerator(rescale=1./255)
        val_data_gen = ImageDataGenerator(rescale=1./255)

        train_data = train_data_gen.flow_from_directory(
            train_dir, target_size=(image_size, image_size),
            color_mode='rgb', batch_size=batch_size,
            class_mode='categorical', shuffle=True) 

        validation_data = val_data_gen.flow_from_directory(
            val_dir, target_size=(image_size, image_size),
            color_mode='rgb', batch_size=batch_size,
            class_mode='categorical', shuffle=False)

        # AI model definition
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same',
            input_shape=(image_size, image_size, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(opt, loss='categorical_crossentropy',
            metrics=['accuracy'])

        model.summary()

        history = model.fit(train_data, epochs=epochs, validation_data=validation_data, verbose=1)

        score = model.evaluate(validation_data)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,4))
        self.plot_loss(history,axL)
        self.plot_acc(history,axR)

        validation_data.reset()
        validation_data.shuffle = False
        validation_data.batch_size = 1
        predicted = model.predict(validation_data, steps=validation_data.n)
        predicted_classes = np.argmax(predicted, axis=-1)
        cm = confusion_matrix(validation_data.classes,predicted_classes)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df = pd.DataFrame(cm, index=labels, columns=labels)
        
        os.makedirs(f'{backup_dir}/confusion_matrix', exist_ok=True)
        plt.figure()
        sns.heatmap(df, annot=True, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xlim([0.0, len(validation_data.class_indices)])
        plt.ylim([0.0, len(validation_data.class_indices)])
        if 'confusion_matrix_1.jpg' in f'{self.workdirectory}/cfar/all/confusion_matrix':
            flag = True
            matrix_number = 2
            while flag:
                if f'confusion_matrix_{matrix_number}.jpg' in f'{self.workdirectory}/cfar/all/confusion_matrix':
                    matrix_number += 1
                else:
                    plt.savefig(f'{self.workdirectory}/cfar/all/confusion_matrix/confusion_matrix_{matrix_number}.jpg')
                    flag = False
        else:
            plt.savefig(f'{self.workdirectory}/cfar/all/confusion_matrix/confusion_matrix_1.jpg')
        plt.show()

        # Save model
        save_model_path = os.path.join(backup_dir, 'my_model.h5')
        model.save(save_model_path)

    def plot_loss(self, fit, axL):
        #損失をプロット
        axL.plot(fit.history['loss'],label="loss for training")
        axL.plot(fit.history['val_loss'],label="loss for validation")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='best')

    def plot_acc(self, fit,axR):
        #精度をプロット
        axR.plot(fit.history['accuracy'],label="loss for training")
        axR.plot(fit.history['val_accuracy'],label="loss for validation")
        axR.set_title('model accuracy')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc='best')
    
    def return_target(self,train_dir, val_dir):
        #再学習のため、検証用に分離した画像を元に戻し、再度分離できるようにする。
        val_list = os.listdir(val_dir)
        for i in val_list:
            img_list = os.listdir(f'{val_dir}/{i}')
            for img in img_list:
                shutil.move(f'{val_dir}/{i}/{img}', f'{train_dir}/{i}/{img}')
    
    def merge_detect(self, train_dir):
        #detected内で目視で正しく分類したものをtrainに戻す。
        detected_list = os.listdir(f'{self.workdirectory}/cfar/all/detected')
        for i in detected_list:
            img_list = os.listdir(f'{self.workdirectory}/cfar/all/detected/{i}')
            for img in img_list:
                shutil.move(f'{self.workdirectory}/cfar/all/detected/{i}/{img}', f'{train_dir}/{i}/{img}')
    
    def model_cp(self, model_dir):
        #既に別のOBSでモデルを作成している場合、作業中のフォルダにモデルをコピーする。
        if 'model' not in os.listdir(f'{self.workdirectory}/cfar/all'):
            shutil.copytree(model_dir, f'{self.workdirectory}/cfar/all/model')

    
    def sep_cnn_noise(self, test_img_list, detected_dir, model_name, days, all_result):
        #fig_convertから実際に仕分けする
        img_list = glob.glob(f'{test_img_list}/{days}*.jpg')
        print(days)
        target_list = []
        model = load_model(f'{self.workdirectory}/cfar/all/model/{model_name}') 

        for i in range(len(img_list)):
            img_path = load_img(img_list[i], target_size=(128,128))
            img_array = img_to_array(img_path)
            img = img_to_array(img_array)/255
            img = img[None,]
            target_list.append(img)

        labels = []
        with open(f'{self.workdirectory}/cfar/all/model/labels.txt', 'r', encoding='utf=8') as f:
            for line in f:
                line = line.rstrip()
                labels.append(line)

        result = [0]*len(labels)

        os.makedirs(detected_dir, exist_ok=True)
        
        for i in labels:
            os.makedirs(f'{detected_dir}/{i}', exist_ok=True)

        for target in tqdm(range(len(target_list))):
            pred = model.predict(target_list[target], batch_size=1)
            score = np.max(pred)
            pred_label = labels[np.argmax(pred[0])]
            result[labels.index(pred_label)] += 1
            all_result[labels.index(pred_label)] += 1
            for i in labels:
                if pred_label == i:
                    shutil.move(img_list[target],f'{detected_dir}/{i}')
        
        print(result)

        return all_result


    def make_peaktxt(self, filelist, dir_list):
        for dir in dir_list:
            fig_dir = f'{self.workdirectory}/cfar/all/sep_fig/{dir}'
            fig_list = os.listdir(fig_dir)

            for day in filelist:
                peak_idx = []
                name = (re.search(r"\d{6}", day)).group()

                for fig in fig_list:
                    if fig[:6] == name:
                        peak_idx.append(fig)
                
                day_call_list = []
                idx_list = []
                #1日毎のパラメータ、callの時間、何番目の要素かをテキストファイルで出力する
                with open(f'{self.workdirectory}/cfar/{name}/peak.txt', 'r' ,encoding='utf-8', newline='\n') as f:
                    for line in f:
                        day_call_list.append(line)
                day_call_list = day_call_list[2:]
                for i in range(len(day_call_list)):
                    day_call_list[i] = day_call_list[i].split(' ')

                with open(f'{self.workdirectory}/cfar/{name}/peak_{dir}.txt', 'w',encoding='utf-8', newline='\n') as f:
                    f.write(f'検出数：{str(len(peak_idx))} num_train={self.num_train} num_guard={self.num_guard} rate_fa={self.rate_fa}\nfmin={self.fmin} fmax={self.fmax} fpass={self.fpass} fstop={self.fstop} gpass={self.gpass} gstop={self.gstop}\n')
                    for i in peak_idx:
                        idx_list.append(int(i[7:-8]))
                    idx_list.sort()

                    for idx in idx_list:
                        f.write(str(day_call_list[idx][0]) + ' ' + str(day_call_list[idx][1]))


    def noise_graph(self, filelist,new_or, day_scale, count_scale):
        os.makedirs(f'{self.workdirectory}/fig/{new_or}', exist_ok=True)
        group_dir = f'{self.workdirectory}/cfar/all/sep_fig'
        group_list = os.listdir(group_dir)

        fig_name_list = [[[] for j in range(len(group_list))] for i in range(len(filelist))]

        for group in range(len(group_list)):
            fig_list = os.listdir(f'{group_dir}/{group_list[group]}')
            count = 0
            for day in filelist:
                yymmdd = (re.search(r"\d{6}", day)).group()
                for fig in fig_list:
                    if yymmdd in fig:
                        fig_name_list[count][group].append(fig)
                count += 1
        
        for group in range(len(group_list)):
            n_fig = []
            for i in range(len(fig_name_list)):
                n_fig.append(len(fig_name_list[i][group]))
            plt.figure()
            plt.plot(n_fig)
            plt.title(f'{group_list[group]}')
            plt.ylabel('num')
            plt.xlabel('day')
            plt.xticks(count_scale,day_scale)
            plt.savefig(f'{self.workdirectory}/fig/{new_or}/{group_list[group]}.jpg')
            plt.show()



#----------------------------------------------------------------------------------------------------#
class NCallAnalysis(Cfar):
    #n_callを修正し、再度モデルで分類を行う。

    def __init__(self, workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa):
        super().__init__(workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa)
    
    def make_fig(self, wav, data_dir, callname):
        #画像を作成する。
        c = np.abs(librosa.stft(wav,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
        C, phase = librosa.magphase(c)
        Cdb = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        librosa.display.specshow(data=Cdb, sr=self.sr,fmin=10, fmax=self.fmax, x_axis='time', y_axis='hz')
        plt.ylim(10, self.fmax)
        plt.axis('off')
        plt.savefig(f'{data_dir}/{callname}')
        plt.cla() #メモリが足りない場合これを試す

        img = Image.open(f'{data_dir}/{callname}')
        img = img.convert('RGB')
        img_resize = img.resize((100, 100)) #画像サイズ100*100
        img_resize.save(f'{data_dir}/{callname}')

    def up_call(self, wav):
        #音声ファイル全体に絶対値の最小値が5e-6を上回るまで10倍する。
        m_wav = np.min(np.abs(wav))
        while m_wav < 5e-6:
            wav = wav*10
            m_wav = np.min(np.abs(wav))
        return wav

    def plus_noise(self,wav):
        y = self.up_call(wav=wav)
        A = np.max(np.abs(y))/2
        x = np.random.rand(round(len(y)))*A/2
        for i in range(len(y)):
            if np.abs(y[i]) < A:
                if y[i] < 0:
                    y[i] -= x[i]
                else:
                    y[i] += x[i]
        return y

    
    def up_ncall(self, n_call_list_dir, mode):
        #上記処理を実行する.
        n_call_list = os.listdir(f'{self.workdirectory}/{n_call_list_dir}')
        zero_call_list = copy.deepcopy(n_call_list)
        zero_call_list = [i.rstrip('.jpg.jpg') for i in zero_call_list]
        for i in range(len(zero_call_list)):
            zero_call_list[i] = zero_call_list[i].split('_')
            wav_idx = str(zero_call_list[i][1])
            while len(wav_idx) < 6:
                wav_idx = '0' + wav_idx
            zero_call_list[i] = zero_call_list[i][0] + '_' + wav_idx

        data_dir = f'{self.workdirectory}/cfar/all/n_call_fig'
        os.makedirs(data_dir, exist_ok=True)

        for i in tqdm(range(len(zero_call_list))):
            callday = re.search(r'\d{6}',zero_call_list[i]).group()
            y, sr = librosa.audio.load(f'{self.workdirectory}/cfar/{callday}/split/{zero_call_list[i]}.wav', sr=self.sr)
            if mode == 'volumeup':
                y = self.up_call(wav=y)
            elif mode == 'plus_noise':
                y = self.plus_noise(wav=y)
            self.make_fig(wav=y,data_dir=data_dir, callname=n_call_list[i])
            os.remove(f'{self.workdirectory}/{n_call_list_dir}/{n_call_list[i]}')
      
    def del_ncall(self, path, check):
        if check:
            shutil.rmtree(path)
            os.makedirs(path)
            print('deleted')
    
    def move_ncall(self, data_dir, target_dir):
        #分類した画像をsep_figに戻す
        ddir = os.listdir(data_dir)
        tdir = os.listdir(target_dir)
        self.del_ncall(path=f'{target_dir}/n_call', check=True)
        for i in range(len(ddir)):
            if ddir[i] == tdir[i]:
                fig_list = os.listdir(f'{data_dir}/{ddir[i]}')
                for j in fig_list:
                    shutil.move(f'{data_dir}/{ddir[i]}/{j}', f'{target_dir}/{tdir[i]}')
    

#----------------------------------------------------------------------------------------------------#


class Analysis(Cfar):
    def __init__(self, workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa):
        super().__init__(workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa)        

    def get_call(self,day):
        name = (re.search(r"\d{6}", day)).group()
        peak_idx = []
        peak_call_idx = []

        with open(f'{self.workdirectory}/cfar/{name}/peak_call.txt', 'r' ,encoding='utf-8', newline='\n') as f:
            for line in f:
                peak_call_idx.append(line)
        peak_call_idx = peak_call_idx[2:]
        for i in range(len(peak_call_idx)):
            peak_call_idx[i] = peak_call_idx[i].split(' ')

        with open(f'{self.workdirectory}/cfar/{name}/peak.txt', 'r' ,encoding='utf-8', newline='\n') as f:
            for line in f:
                peak_idx.append(line)
        peak_idx = peak_idx[2:]
        for i in range(len(peak_idx)):
            peak_idx[i] = peak_idx[i].split(' ')

        return peak_call_idx, peak_idx, name

    #中心周波数を取得する
    def centroid(self, handle_day, wav_idx):
        y, sr_none = librosa.audio.load(f'{self.workdirectory}/cfar/{handle_day}/split/{handle_day}_{wav_idx}.wav', sr=self.sr)

        num = 1
        num_count = 0
        while num > 0.0001:
            
            if num_count == 0:
                adjust_fpass = np.array([10,40])
            else:
                adjust_fpass = np.array([cent.T[cent_peak_idx]-5,cent.T[cent_peak_idx]+5])
            if adjust_fpass[0] < self.fstop[0]:
                break
            if adjust_fpass[1] > self.fstop[1]:
                break
            #バンドパスフィルタを適用する
            data = self.bandpass(x=y, freqpass=adjust_fpass, freqstop=self.fstop, gpass=self.gpass, gstop=self.gstop)
            S, phase = librosa.magphase(librosa.stft(y=data, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
            cent = librosa.feature.spectral_centroid(S=S, sr=self.sr,n_fft=3,hop_length=2,window='hann')
            times = librosa.times_like(cent,sr=self.sr)
            
            if num_count == 0:
                cent_peak_idx = len(cent.T)//2
            
            if num_count > 0:
                num = hz - cent.T[cent_peak_idx]
            hz = cent.T[cent_peak_idx]
            
            num_count += 1
            if cent.T[cent_peak_idx] < 7.5:
                fp = np.array([5.5, 10.5])
            elif cent.T[cent_peak_idx] > 47.5:
                fp = np.array([44.5,49.5])
            else:
                fp = np.array([cent.T[cent_peak_idx]-2.5,cent.T[cent_peak_idx]+2.5])
            hzdata = self.bandpass(x=y, freqpass=fp, freqstop=self.fstop, gpass=self.gpass, gstop=self.gstop)
            
            temp =len(hzdata)//3 + np.argmax(np.abs(hzdata[len(hzdata)//3:2*len(hzdata)//3+1]))
            cent_peak_idx = round(len(times)*temp/len(hzdata))
    
        cent_num = [times[cent_peak_idx], cent.T[cent_peak_idx]]

        return cent_num


    def run_centroid(self, filelist):
        os.makedirs(f'{self.workdirectory}/analysis', exist_ok=True)
        centroid_list = []
        if 'temp_centroid_list.txt' not in os.listdir(f'{self.workdirectory}/analysis'):
            with open(f'{self.workdirectory}/analysis/temp_centroid_list.txt', 'wb') as f:
                    pickle.dump(centroid_list, f)
        
        with open(f'{self.workdirectory}/analysis/temp_centroid_list.txt', 'rb') as f:
            temp_list = pickle.load(f)

        for day in tqdm(range(len(filelist)-len(temp_list))):
            peak_call_idx, peak_idx, name = self.get_call(filelist[len(temp_list)])
            centroid_list = []

            for call in peak_call_idx:
                for idx in range(len(peak_idx)):
                    if call[1] in peak_idx[idx][1]:
                        wav_idx = str(idx)
                        while len(wav_idx) < 6:
                            wav_idx = '0' + wav_idx

                        cent_num = self.centroid(handle_day=name, wav_idx=wav_idx)
                        centroid_list.append(cent_num)
                        break
            

            temp_list.append(centroid_list)
            with open(f'{self.workdirectory}/analysis/temp_centroid_list.txt', 'wb') as f:
                pickle.dump(temp_list, f)
        
        return temp_list

    
    def one_cent_fig(self, handle_day, number):
        #callひとつの中心周波数の特定過程の画像を出力する
        y, sr = librosa.audio.load(f'{self.workdirectory}/cfar/{handle_day}/split/{handle_day}_{number}.wav', sr=self.sr)
        num = 1
        num_count = 0
        while num > 0.0001:
            
            if num_count == 0:
                adjust_fpass = np.array([10,40])
            else:
                adjust_fpass = np.array([cent.T[cent_peak_idx]-5,cent.T[cent_peak_idx]+5])
            #バンドパスフィルタを適用する
            data = self.bandpass(x=y, freqpass=adjust_fpass, freqstop=self.fstop, gpass=self.gpass, gstop=self.gstop)
            S, phase = librosa.magphase(librosa.stft(y=data, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
            cent = librosa.feature.spectral_centroid(S=S, sr=self.sr,n_fft=3,hop_length=2,window='hann')
            times = librosa.times_like(cent,sr=self.sr)
            
            if num_count == 0:
                cent_peak_idx = len(cent.T)//2
            
            if num_count > 0:
                num = hz - cent.T[cent_peak_idx]
            hz = cent.T[cent_peak_idx]
            
            num_count += 1
            print(hz)

            hzdata = self.bandpass(x=y, freqpass=np.array([cent.T[cent_peak_idx]-2.5,cent.T[cent_peak_idx]+2.5]), freqstop=self.fstop, gpass=self.gpass, gstop=self.gstop)
            
            temp =len(hzdata)//3 + np.argmax(np.abs(hzdata[len(hzdata)//3:2*len(hzdata)//3+1]))
            cent_peak_idx = round(len(times)*temp/len(hzdata))

            S, phase = librosa.magphase(librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
            Cdb = librosa.amplitude_to_db(np.abs(S), ref=np.max)

            fig, ax = plt.subplots()
            librosa.display.specshow(data=Cdb, sr=sr,fmin=10,fmax=self.fmax, x_axis='s', y_axis='hz', ax=ax)
            ax.plot(times, cent.T, label='Spectral centroid', color='g')
            ax.plot(times[cent_peak_idx], cent.T[cent_peak_idx],marker='.', color='blue',markersize =20)
            ax.set_ylim(10,self.fmax)
            ax.legend(loc='upper right')
            ax.set(title='spectrogram')
            ax.set_xticks([times[0],times[100],times[200],times[300]])
            ax.set_xticklabels([0,1,2,3])
            os.makedirs(f'{self.workdirectory}/centroid',exist_ok=True)
            fig.savefig(f'{self.workdirectory}/centroid/{handle_day}_{number}_{num_count}.jpg')
        



    #IPIを取得する
    def ipi(self, all_time_list, limsec):

        date_list = os.listdir(f'{self.workdirectory}/cfar')
        if 'all' in date_list:
            date_list.remove('all')

        if  'pick_list.txt' in date_list:
            date_list.remove('pick_list.txt')
        
        date_list = list(map(int, date_list))
        date_list.sort()
        date_list = list(map(str,date_list))


        ipi_list = []
        ipi_all_list = []
        count = 0
        for day in tqdm(range(len(date_list))):
            y, sr_none = librosa.audio.load(f'{self.workdirectory}/cfar/{date_list[day]}/split/{date_list[day]}_000000.wav', sr=self.sr)
            S, phase = librosa.magphase(librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
            cent = librosa.feature.spectral_centroid(S=S, sr=self.sr,n_fft=3,hop_length=2,window='hann')
            times = librosa.times_like(cent,sr=self.sr)
            peak_call_idx, peak_idx, name = self.get_call(date_list[day])
            ipi_day_list = []
            ipi_all_day_list = []

            if len(peak_call_idx)>=2:
                time_list = all_time_list[count:len(peak_call_idx)+count]
                time_list = [int((i/np.max(times))*self.sr*3-300) for i in time_list]

                for i in range(len(peak_call_idx)-1):
                    ipi_num = int(int(peak_call_idx[i+1][1])+time_list[i+1]) - int(int(peak_call_idx[i][1])+time_list[i])
                    if ipi_num <= limsec*self.sr:
                        ipi_day_list.append(ipi_num)
                        ipi_all_day_list.append(ipi_num)
                    else:
                        ipi_all_day_list.append(ipi_num)

                if len(ipi_day_list) > 0:
                    ipi_list.append(ipi_day_list)
                    ipi_all_list.append(ipi_all_day_list)
                else:
                    ipi_list.append([])
                    ipi_all_list.append(ipi_all_day_list)
                
                count += len(peak_call_idx)
            else:
                ipi_list.append([])
                ipi_all_list.append([])
            

        return ipi_list, ipi_all_list
    
    def callma_graph(self, list_row, count_scale, day_scale):
        #1日辺りのcallと30日の移動平均線
        day_list = [len(list_row[i]) for i in range(len(list_row))]
        dd = pd.Series(day_list).rolling(30).mean()
        plt.plot(day_list)
        plt.plot(dd)
        plt.title('call')
        plt.xlabel('day')
        plt.ylabel('num')
        plt.xticks(count_scale,day_scale)
        plt.savefig(f'{self.workdirectory}/fig/call_ave.jpg')
    
    def ipi_histgram(self, ipi_row, bins):
        #ipiのヒストグラムを表示
        s_ipi_list = []
        for day in ipi_row:
            for i in day:
                s_ipi_list.append(i/self.sr)
        s_ipi_list.sort()
        plt.hist(s_ipi_list, bins=bins)
        plt.title('ipi_histgram')
        plt.xlabel('Second')
        plt.ylabel('num')
        plt.savefig(f'{self.workdirectory}/fig/ipi_histgram.jpg')

    def ipima_graph(self, ipi_row, count_scale, day_scale):
        #ipiの1日辺りの平均値と30日移動平均線
        ipi_average = []

        for i in range(len(ipi_row)):
            if len(ipi_row[i]) > 0:
                a = [j/self.sr for j in ipi_row[i]]
                ipi_average.append(np.average(a))
            elif i == 0:
                a = [j/self.sr for j in ipi_row[i+1]]
                ipi_average.append(np.average(a))
            else:
                if i > 2:
                    ipi_average.append(np.average(ipi_average))
        
        plt.plot(ipi_average)
        ma = pd.Series(ipi_average).rolling(30).mean()
        plt.plot(ma)
        plt.xticks(count_scale,day_scale)
        plt.ylabel('ipi')
        plt.xlabel('day')
        plt.title('ipi')
        plt.savefig(f'{self.workdirectory}/fig/ipi_average.jpg')

        return ipi_average
    
    def cent_histgram(self, all_cent_list, bins):
        all_cent_list.sort()
        al = list(map(float, all_cent_list))
        ar = [round(i, 1) for i in al]

        plt.hist(ar, bins=bins)
        plt.title('centroid_histgram')
        plt.xlabel('Hz')
        plt.ylabel('num')
        plt.savefig(f'{self.workdirectory}/fig/centroid_histgram.jpg')

    def centroid_timescales_graph(self,count_scale,day_scale):
        #中心周波数の時系列グラフを作成する。
        with open(f'{self.workdirectory}/analysis/centroid_list.txt', 'rb') as f:
            list_row = pickle.load(f)
        cent_list = []
        for day in range(len(list_row)):
            day_calls = []
            for calls in range(len(list_row[day])):
                day_calls.append(list_row[day][calls][1])
            if len(day_calls) > 0:
                cent_list.append(np.average(day_calls))
            else:
                cent_list.append(21.25)

        plt.figure()
        cent_ma = pd.Series(cent_list).rolling(30).mean()
        plt.plot(cent_list)
        plt.plot(cent_ma)
        plt.title('day_centroid')
        plt.xlabel('day')
        plt.ylabel('centroid')
        plt.xticks(count_scale,day_scale)
        plt.savefig(f'{self.workdirectory}/fig/day_centroid.jpg')
        plt.show()
        
        return cent_list
    
    def ma_graph(self, day_list, ipi_average, day_cent_list,count_scale,day_scale):
        ma_range = 30
        nan_list = [np.nan for i in range(ma_range-1)]
        day_ma = pd.Series(day_list).rolling(ma_range).mean()[ma_range-1:]
        daymin = np.min(day_ma)
        day_ma = [i-daymin for i in day_ma]
        daymax = np.max(day_ma)
        day_ma = [i/daymax for i in day_ma]
        day_ma = nan_list+day_ma
        ipi_ma = pd.Series(ipi_average).rolling(ma_range).mean()[ma_range-1:]
        ipimin = np.min(ipi_ma)
        ipi_ma = [i-ipimin for i in ipi_ma]
        ipimax = np.max(ipi_ma)
        ipi_ma = [i/ipimax for i in ipi_ma]
        ipi_ma = nan_list+ipi_ma
        cent_ma = pd.Series(day_cent_list).rolling(ma_range).mean()[ma_range-1:]
        centmin = np.min(cent_ma)
        cent_ma = [i-centmin for i in cent_ma]
        centmax = np.max(cent_ma)
        cent_ma = [i/centmax for i in cent_ma]
        cent_ma = nan_list + cent_ma

        plt.plot(day_ma, label='call_ma')
        plt.plot(ipi_ma, label='ipi_ma')
        plt.plot(cent_ma, label='cent_ma')
        plt.title('call_num,ipi,centroid')
        plt.xlabel('day')
        plt.ylabel('nomalized')
        plt.xticks(count_scale,day_scale)
        plt.legend(loc='upper right')
        plt.savefig(f'{self.workdirectory}/fig/call_ipi_cent_ma.jpg')
        plt.show()

    def cent_day_graph(self, filelist):
        #1日毎の中心周波数の分布をグラフ化
        os.makedirs(f'{self.workdirectory}/centroid/cent', exist_ok=True)

        one_dayscale = list(range(0,25,1))
        one_countscale = [one_dayscale[i]*3600*self.sr for i in range(len(one_dayscale))]

        with open(f'{self.workdirectory}/analysis/centroid_list.txt', 'rb') as f:
            list_row = pickle.load(f)

        max_list = []
        min_list = []
        for day in range(len(filelist)):
            maxmin_list = [float(list_row[day][i][1]) for i in range(len(list_row[day]))]
            if len(maxmin_list) > 0:
                max_list.append(np.max(maxmin_list))
                min_list.append(np.min(maxmin_list))
        cent_max = np.max(max_list)
        cent_min = np.min(min_list)
        plt.figure()
        for day in range(len(filelist)):
            peak_call_idx, peak_idx, name = self.get_call(filelist[day])
            peaks = [int(i[1]) for i in peak_call_idx]
            cent_list = [float(list_row[day][i][1]) for i in range(len(list_row[day]))]
            
            plt.scatter(peaks, cent_list, s=10)
            plt.title(f'{name}_centroid')
            plt.ylabel('centroid')
            plt.xlabel('hour')
            plt.xticks(one_countscale,one_dayscale)
            plt.ylim(cent_min, cent_max)
            plt.savefig(f'{self.workdirectory}/centroid/cent/{name}_cent.jpg')
            plt.cla()
        
        #1日変化の1時間ごとの月合計をグラフ化
        os.makedirs(f'{self.workdirectory}/centroid/cent/month', exist_ok=True)
        h_dayscale = list(range(0,70,10))
        month_cent_list = [[] for i in range(24)]
        month_peak_list = [[] for i in range(24)]
        peak_call_idx, peak_idx, name = self.get_call(filelist[0])
        new_name = name[:4]
        for day in range(len(filelist)):
            peak_call_idx, peak_idx, name = self.get_call(filelist[day])
            peaks = [int(i[1]) for i in peak_call_idx]
            cent_list = [float(list_row[day][i][1]) for i in range(len(list_row[day]))]
            if name[:4] == new_name:
                for peak in range(len(peaks)):
                    for hour in range(len(month_cent_list)):
                        if peaks[peak] < 3600*200*(hour+1):
                            month_cent_list[hour].append(cent_list[peak])
                            month_peak_list[hour].append(peaks[peak])
                            break
            else:
                
                for hour in range(len(month_cent_list)):
                    h_countscale = [h_dayscale[i]*60*self.sr+(hour*3600*self.sr) for i in range(len(h_dayscale))]
                    plt.scatter(month_peak_list[hour], month_cent_list[hour], s=10)
                    plt.title(f'{new_name}_{hour}-{hour+1}_centroid')
                    plt.ylabel('centroid')
                    plt.xlabel(f'{hour}-{hour+1}')
                    plt.xticks(h_countscale,h_dayscale)
                    plt.ylim(cent_min, cent_max)
                    plt.savefig(f'{self.workdirectory}/centroid/cent/month/{new_name}_{hour}-{hour+1}_cent.jpg')
                    plt.cla()
                all_cent = []
                all_peak = []
                for i in range(len(month_cent_list)):
                    for j in range(len(month_cent_list[i])):
                        all_cent.append(month_cent_list[i][j])
                        all_peak.append(month_peak_list[i][j])
                plt.scatter(all_peak, all_cent, s=5)
                plt.title(f'{new_name}_centroid')
                plt.ylabel('centroid')
                plt.xlabel('hour')
                plt.xticks(one_countscale,one_dayscale)
                plt.ylim(cent_min, cent_max)
                plt.savefig(f'{self.workdirectory}/centroid/cent/month/{new_name}_cent.jpg')
                plt.cla()

                
                new_name = name[:4]
                month_cent_list = [[] for i in range(24)]
                month_peak_list = [[] for i in range(24)]
                for peak in range(len(peaks)):
                    for hour in range(len(month_cent_list)):
                        if peaks[peak] < 3600*200*(hour+1):
                            month_cent_list[hour].append(cent_list[peak])
                            month_peak_list[hour].append(peaks[peak])
                            break
            
            for hour in range(len(month_cent_list)):
                h_countscale = [h_dayscale[i]*60*self.sr+(hour*3600*self.sr) for i in range(len(h_dayscale))]
                plt.scatter(month_peak_list[hour], month_cent_list[hour], s=10)
                plt.title(f'{new_name}_{hour}-{hour+1}_centroid')
                plt.ylabel('centroid')
                plt.xlabel(f'{hour}-{hour+1}')
                plt.xticks(h_countscale,h_dayscale)
                plt.ylim(cent_min, cent_max)
                plt.savefig(f'{self.workdirectory}/centroid/cent/month/{new_name}_{hour}-{hour+1}_cent.jpg')
                plt.cla()
            all_cent = []
            all_peak = []
            for i in range(len(month_cent_list)):
                for j in range(len(month_cent_list[i])):
                    all_cent.append(month_cent_list[i][j])
                    all_peak.append(month_peak_list[i][j])
            plt.scatter(all_peak, all_cent, s=5)
            plt.title(f'{new_name}_centroid')
            plt.ylabel('centroid')
            plt.xlabel('hour')
            plt.xticks(one_countscale,one_dayscale)
            plt.ylim(cent_min, cent_max)
            plt.savefig(f'{self.workdirectory}/centroid/cent/month/{new_name}_cent.jpg')
            plt.cla()

    def ipi_cent_graph(self, filelist):
            with open(f'{self.workdirectory}/analysis/centroid_list.txt', 'rb') as f:
                list_row = pickle.load(f)
            with open(f'{self.workdirectory}/analysis/ipi_all_list.txt', 'rb') as f:
                ipi_all_row = pickle.load(f)

            max_list = []
            min_list = []
            for day in range(len(filelist)):
                maxmin_list = [float(list_row[day][i][1]) for i in range(len(list_row[day]))]
                if len(maxmin_list) > 0:
                    max_list.append(np.max(maxmin_list))
                    min_list.append(np.min(maxmin_list))
            cent_max = np.max(max_list)
            cent_min = np.min(min_list)

            os.makedirs(f'{self.workdirectory}/centroid/ipi_cent', exist_ok=True)

            for day in range(len(list_row)):
                peak_call_idx, peak_idx, name = self.get_call(filelist[day])
                a = [float(list_row[day][i][1]) for i in range(len(list_row[day]))][1:]
                b = [i/self.sr for i in ipi_all_row[day]]
                plt.figure()
                plt.scatter(b, a, s=5)
                plt.title(f'{name}_centipi')
                plt.ylabel('centroid')
                plt.xlabel('ipi')
                plt.xlim(0,45)
                plt.ylim(cent_min, cent_max)
                plt.savefig(f'{self.workdirectory}/centroid/ipi_cent/{name}_centipi.jpg')
    
    def ipi_day_graph(self, filelist, ipi_all_row):
        os.makedirs(f'{self.workdirectory}/centroid/ipi', exist_ok=True)
        one_dayscale = list(range(0,25,1))
        one_countscale = [one_dayscale[i]*3600*self.sr for i in range(len(one_dayscale))]
        plt.figure()
        for day in range(len(ipi_all_row)):
            peak_call_idx, peak_idx, name = self.get_call(filelist[day])
            peaks = [int(i[1]) for i in peak_call_idx][1:]
            ipis = [i/self.sr for i in ipi_all_row[day]]

            plt.scatter(peaks, ipis, s=3)
            plt.xticks(one_countscale,one_dayscale)
            plt.ylim(0,45)
            plt.title(f'{name}_ipi')
            plt.xlabel('hour')
            plt.ylabel('ipi')
            plt.savefig(f'{self.workdirectory}/centroid/ipi/{name}_ipi.jpg')
            plt.cla()
    
    def detect_ipi_sequence(self, ipi_all_row, seq_min_num):
        ipi_sequence = [[] for i in range(len(ipi_all_row))]
        for i in range(len(ipi_all_row)):
            ipi_one_seq = []
            if len(ipi_all_row[i]) > 0:
                for j in range(len(ipi_all_row[i])):
                    if ipi_all_row[i][j] < 45*self.sr:
                        ipi_one_seq.append([j+1, ipi_all_row[i][j]])
                    else:
                        if len(ipi_one_seq) >= seq_min_num:
                            ipi_sequence[i].append(ipi_one_seq)
                        ipi_one_seq = []
        return ipi_sequence
    
    def detect_singlecall(self,ipi_all_row, single_ipi, count_scale,day_scale):
        single_calls = []
        single_ave = []
        for i in range(len(ipi_all_row)):
            day_num = len(ipi_all_row[i])
            day_single = 0
            day_single_list = []
            for j in range(len(ipi_all_row[i])):
                if single_ipi[0]*self.sr <= ipi_all_row[i][j] <= single_ipi[1]*self.sr:
                    day_single_list.append(ipi_all_row[i][j]/self.sr)
                    day_single += 1
            if day_num == 0:
                single_calls.append(0)
            else:
                single_calls.append((day_single/day_num)*100)
            single_ave.append(np.mean(day_single_list))
        
        dd = pd.Series(single_calls).rolling(30).mean()
        plt.figure()
        plt.plot(single_calls)
        plt.plot(dd)
        plt.title(f'single_calls_{single_ipi[0]}-{single_ipi[1]}')
        plt.ylabel('%')
        plt.xlabel('day')
        plt.xticks(count_scale,day_scale)
        plt.savefig(f'{self.workdirectory}/fig/single_calls.jpg')
        plt.close()

        
        plt.figure()
        plt.plot(single_ave)
        plt.title(f'single_ave_{single_ipi[0]}-{single_ipi[1]}')
        plt.ylabel('ipi_ave')
        plt.xlabel('day')
        plt.ylim(single_ipi[0],single_ipi[1])
        plt.xticks(count_scale,day_scale)
        plt.savefig(f'{self.workdirectory}/fig/single_ave.jpg')
        plt.close()

    
    def temp_cent(self, filelist, list_row):
        #中心周波数の修正用。修正が無ければそのまま通る。
        for day in range(len(filelist)):
            peak_call_idx, peak_idx, name = self.get_call(filelist[day])
            if len(peak_call_idx) != len(list_row[day]):
                temp_list = []
                for call in tqdm(range(len(peak_call_idx))):
                    for idx in range(len(peak_idx)):
                        if peak_call_idx[call][1] in peak_idx[idx][1]:
                            wav_idx = str(idx)
                            while len(wav_idx) < 6:
                                wav_idx = '0' + wav_idx

                            cent_num = self.centroid(handle_day=name, wav_idx=wav_idx)
                            temp_list.append(cent_num)
                            break
                print(name+'repaired')
                list_row[day] = temp_list
            else:
                print(name+'clear')
        
        for day in range(len(filelist)):
            peak_call_idx, peak_idx, name = self.get_call(filelist[day])
            if len(peak_call_idx) != len(list_row[day]):
                print(filelist[day])
        
        return list_row
    
    def all_obs_ipi(self, low_sec, high_sec, path):
        with open(f'{path}/analysis/ipi_all_list.txt', 'rb') as f:
            ipi_all_row = pickle.load(f)
        low = 0
        mid = 0
        high = 0
        for day in ipi_all_row:
            for call in day:
                if call <= low_sec*self.sr:
                    low += 1
                elif call > high_sec*self.sr:
                    high += 1
                elif low_sec*self.sr < call <=high_sec*self.sr:
                    mid += 1
        return low,mid,high


#----------------------------------------------------------------------------------------------------#


#使ってないです
class Clustering(Analysis):
    #クラスタリングを行う

    def __init__(self, workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa):
        super().__init__(workdirectory, datadirectory , sr, n_fft, hop_length, fmin, fmax, fpass, fstop, gpass, gstop, num_train, num_guard, rate_fa)

    

    def cluster_qw_sent(self, conv_dir, cluster_number):
        #1年分の全ての画像を一括でクラスタリングをする

        group_dir = f'{self.workdirectory}/cfar/all/fig_group'
        
        #memmapの作成
        image_list = os.listdir(conv_dir)

        if os.path.exists(f'{self.workdirectory}/cfar/all/dataset.dat'):
            pass
        else:
            fp = np.memmap(f'{self.workdirectory}/cfar/all/dataset.dat', dtype='float32',mode='w+', shape=(len(image_list),100*100*3))
            del fp
            for i in range(len(image_list)):
                img = np.array((io.imread(f'{conv_dir}/{image_list[i]}')).astype('float32')).reshape(-1)
                fp = np.memmap(f'{self.workdirectory}/cfar/all/dataset.dat', dtype='float32',mode='r+', shape=(len(image_list),100*100*3))
                fp[i] = img
                del fp
        
        feature = np.memmap(f'{self.workdirectory}/cfar/all/dataset.dat', dtype='float32',mode='r', shape=(len(image_list),100*100*3))
        
        #クラスタリング開始
        model = MiniBatchKMeans(n_clusters=cluster_number,batch_size=4096).fit(feature)
        print('学習済み')

        labels = model.labels_
        print('グループ作成中')

        #クラスタリングしたものを元にディレクトリを作成、中に該当する画像をコピーする
        for label, path in zip(labels, os.listdir(conv_dir)):
            os.makedirs(f'{group_dir}/{label}', exist_ok=True)
            shutil.copyfile(f"{conv_dir}/{path}", f"{group_dir}/{label}/{path.replace('.jpg', '',1)}")

                

    def cluster_analysis(self, sr):
        #各グループをx軸を1時間ごとの時間、y軸を1年間の日付、z軸をその時間の鳴音の個数として3Dグラフの作成用データを作成する
        os.makedirs(f'{self.workdirectory}/cfar/all/group_sum',exist_ok=True)
        if len(os.listdir(f'{self.workdirectory}/cfar/all/group_sum')) == 0:
            #各グループの鳴音の分布を調べる
            group_list = os.listdir(f'{self.workdirectory}/cfar/all/fig_group')
            
            #全日付を取得
            date_list = os.listdir(f'{self.workdirectory}/cfar')
            if 'all' in date_list:
                date_list.remove('all')

            if  'pick_list.txt' in date_list:
                date_list.remove('pick_list.txt')
            
            date_list = list(map(int, date_list))
            date_list.sort()
            date_list = list(map(str,date_list))
            

            #グループごとに繰り返し処理
            for path in group_list:
                group = os.listdir(f'{self.workdirectory}/cfar/all/fig_group/{path}')

                year_list = []

                #グループ内の鳴音リストを取得
                fig_list = []
                for i in group:
                    fig = i.rstrip('.jpg').split('_')
                    fig_list.append(fig)       
                
                for date in date_list:
                    #鳴音のindex番号
                    call_list = []
                    for call in range(len(fig_list)):
                        if date == fig_list[call][0]:
                            call_list.append(fig_list[call][1])

                    #鳴音の時間
                    peaks = []
                    with open(f'{self.workdirectory}/cfar/{date}/peak.txt', 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.rstrip()
                            peaks.append(line)
                        peaks = peaks[2:]
                    peaks = [i.split(' ') for i in peaks]

                    group_call_list = []
                    for i in call_list:
                        group_call_list.append(int(peaks[int(i)][1]))
                    
                    day_list = []
                    for i in range(24):
                        hour_list = []
                        for j in group_call_list:
                            if i*3600*sr <= j < (i+1)*3600*sr:
                                hour_list.append(j)
                        day_list.append(hour_list)
                    
                    day_array = np.array([len(i) for i in day_list])
                    
            
                    year_list.append(day_array)
                
                np.save(f'{self.workdirectory}/cfar/all/group_sum/{path}', arr=year_list)

    
    def all_3d_graph(self,sr):
        #全日付を取得
        if 'all.npy' not in os.listdir(f'{self.workdirectory}/cfar/all/'):
            date_list = os.listdir(f'{self.workdirectory}/cfar')
            if 'all' in date_list:
                date_list.remove('all')

            if  'pick_list.txt' in date_list:
                date_list.remove('pick_list.txt')
            
            date_list = list(map(int, date_list))
            date_list.sort()
            date_list = list(map(str,date_list))

            year_list = []
            for date in date_list:
                peaks = []
                with open(f'{self.workdirectory}/cfar/{date}/peak.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.rstrip()
                        peaks.append(line)
                    peaks = peaks[2:]
                peaks = [i.split(' ') for i in peaks]

                day_list = []
                for i in range(24):
                    hour_list = []
                    for j in range(len(peaks)):
                        if i*3600*sr <= int(peaks[j][1]) < (i+1)*3600*sr:
                            hour_list.append(int(peaks[j][1]))
                    day_list.append(hour_list)
                
                day_array = np.array([len(i) for i in day_list])
                
            
                year_list.append(day_array)
            
            np.save(f'{self.workdirectory}/cfar/all/all', arr=year_list)
    

    def cluster_ipi(self, filelist):


        ipi_list = []
        count = 0
        for day in filelist:
            print(f'{day} start')
            y, sr_none = librosa.audio.load(day, sr=self.sr)
            S, phase = librosa.magphase(librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.n_fft,window='hann'))
            cent = librosa.feature.spectral_centroid(S=S, sr=self.sr,n_fft=3,hop_length=2,window='hann')
            times = librosa.times_like(cent,sr=self.sr)
            peak_call_idx, peak_idx, name = self.get_call(day)
            ipi_day_list = []
    
    def run_sep_cent(self, filelist):

        date_list = os.listdir(f'{self.workdirectory}/cfar')
        if 'all' in date_list:
            date_list.remove('all')

        if  'pick_list.txt' in date_list:
            date_list.remove('pick_list.txt')
        
        date_list = list(map(int, date_list))
        date_list.sort()
        date_list = list(map(str,date_list))
        
        group_list = os.listdir(f'{self.workdirectory}/cfar/all/fig_group')

        os.makedirs(f'{self.workdirectory}/analysis/sep', exist_ok=True)
            
        for path in group_list:
            group = os.listdir(f'{self.workdirectory}/cfar/all/fig_group/{path}')
            
            fig_list = []
            for i in group:
                fig = i.rstrip('.jpg').split('_')
                fig_list.append(fig)

            
            centroid_list = []

            for day in tqdm(range(len(date_list)), desc=f'group{path}'):
                cent_ipi = []

                for i in range(len(fig_list)):
                    if date_list[day] == fig_list[i][0]:
                        wav_idx=fig_list[i][1]
                        while len(wav_idx) < 6:
                            wav_idx = '0' + wav_idx
                        cent_num = self.centroid(handle_day=fig_list[i][0], wav_idx=wav_idx)
                        cent_ipi.append(cent_num)
                
                centroid_list.append(cent_ipi)
            
            if f'cent_{path}.txt' not in os.listdir(f'{self.workdirectory}/analysis/sep/'):
                with open(f'{self.workdirectory}/analysis/sep/cent_{path}.txt', 'wb') as f:
                    pickle.dump(centroid_list, f)
            
            print(f'group{path} end')
                        



        
    



        



                





            

 
