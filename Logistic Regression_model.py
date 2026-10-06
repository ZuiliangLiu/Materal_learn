#用于处理分类问题
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
raw_df = pd.read_csv("weatherAUS.csv")
# print(raw_df.head())
# print(raw_df.info())

raw_df.dropna(subset=["RainToday","RainTomorrow"],inplace=True)
# print(raw_df.info())

fig=px.histogram(raw_df,x="Location",title="Location vs. Rainy Days",color="RainToday")
fig.show()





