
small = {'dim': 50,
         'input': 2500,
         'hidden': 1000}
med = {'dim': 100,
       'input': 10000,
       'hidden': 5000}
large = {'dim': 150,
         'input': 22500,
         'hidden': 8000}

img_sizes = {'small': small, 'med': med, 'large': large}





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



from arctic import Arctic
arcticDB = Arctic('mongodb+srv://dbUser:chilicki89@cluster0.lubnq.mongodb.net/testOne?retryWrites=true&w=majority')
def read_from_arctic(parent, child):
    subDB = arcticDB[parent]
    item = subDB.read(child)
    return item 



def round_number(num):
    rounded = round(num/10)*10
    return rounded




