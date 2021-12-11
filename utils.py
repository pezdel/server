import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import torch
import matplotlib.pyplot as plt

small = {'size': 50, 'input': 2500, 'hidden':750}
med= {'size': 100, 'input': 10000, 'hidden':3000}
large = {'size': 150, 'input': 22500, 'hidden':7000}
sizes = {'small': small, 'med': med, 'large': large}





#TODO: make this a class? idk but also make option for vae into cnn for member/elite, basic is just vae
basic_currency = ['EURUSD', 'GBPUSD', 'USDJPY', 'NZDUSD', 'AUDUSD']
basic_tf = ['1d', '1w']
basic = {'currency': basic_currency,
         'tf': basic_tf,
         'ws_range': 10,
         'topx': 1}
member_currency = ['BTCUSD', 'SNP', 'USDX', 'ETHUSD']
member_tf = ['4h', '1h']
member = {'currency': basic_currency + member_currency,
          'tf': basic_tf + member_tf,
          'ws_range': 50,
          'topx': 5}
elite_currency = ['stock', 'stock']
elite_tf = ['5m', '15m']
elite = {'currency': basic_currency + member_currency + elite_currency,
         'tf': basic_tf + member_tf + elite_tf,
         'ws_range': 50,
         'topx': 5}
membership_dict = {'basic': basic, 'member': member, 'elite': elite}







