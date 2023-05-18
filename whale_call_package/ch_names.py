'''
waveファイルの名前変更を行う。
obsnumber/yymmdd-000000.wavの形を想定している。
obsnumberはyy+２桁の番号。
'''

data_dir = f'D:/whale/JB17a' #各obsの１つ上のディレクトリまでの絶対パスを記述

import os
import re
import shutil

obs = os.listdir(data_dir)
for i in obs:
    waves = os.listdir(f'{data_dir}/{i}')
    for wave in waves:
        name = (re.search(r"\d{6}", wave)).group()
        shutil.move(f'{data_dir}/{i}/{wave}', f'{data_dir}/{i}/{name}-000000.wav')