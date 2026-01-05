import pandas as pd
df = pd.read_csv("I:\power_forecast\PyWavelets\datasets\Electricity_load_of_Australia.csv", encoding='gbk')
print(df.columns) # 看看第一行中文到底写了什么