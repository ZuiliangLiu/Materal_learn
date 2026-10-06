import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# =========================
# 图表全局设置
# =========================

sns.set_style('darkgrid')

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# =========================
# 读取数据
# =========================

medical_df = pd.read_csv("medical_charges.csv")# 读取数据
# print(medical_df)#打印一整个数据集
# print(medical_df.head())#打印前五行
# print(medical_df.describe())#统计数据集的平均数、最小值等等
# medical_df.info()#查看各个数据的类型
#
# #查看 age 的统计信息
# print(medical_df["age"].describe())

#绘制年龄分布直方图
# fig = px.histogram(
#     medical_df,#读取的数据集
#     x="age",#x轴
#     marginal="box",#箱线图
#     nbins=47,#箱线图的箱子数
#     title="Distribution of Age"#标题
# )
# # 设置柱子之间的间距
# fig.update_layout(bargap=0.1)
#
# # 显示图像
# fig.show()
#
# fig = px.histogram(
#     medical_df,#读取的数据集
#     x="bmi",#x轴
#     marginal="box",#箱线图
#     color_discrete_sequence=["blue"],
#     title="Distribution of BMI (Body Mass Index)"#标题
# )
#
# fig.show()
#
# fig = px.histogram(
#     medical_df,                              # 数据集
#     x="charges",                             # X轴：年度医疗费用
#     marginal="box",                         # 添加箱线图
#     color="smoker",                         # 按是否吸烟进行分组
#     color_discrete_sequence=["green", "grey"],  # 两个类别使用不同颜色
#     title="Annual Medical Charges"           # 图表标题
# )
#
# fig.update_layout(bargap=0.1)                # 柱子之间设置间隔
# fig.show()
#
# fig = px.histogram(medical_df,
#                    x="smoker",
#                    color="sex",
#                    color_discrete_sequence=["green", "grey"],
#                    title = "Smoker")
# fig.show()
#
# fig = px.scatter(
#     medical_df,            # 使用医疗费用数据集
#     x='age',               # X轴 = 年龄
#     y='charges',           # Y轴 = 医疗费用
#     color='smoker',        # 根据是否吸烟使用不同颜色
#     opacity=0.8,           # 散点透明度80%
#     hover_data=['sex'],    # 鼠标悬停时额外显示性别
#     title='Age vs. Charges' # 图表标题
# )
#
# fig.update_traces(marker_size=5)  # 设置散点大小
# fig.show()                        # 显示图表
#
# print(medical_df.charges.corr(medical_df.age))#计算各个值与charges的相关性
# print(medical_df.charges.corr(medical_df.bmi))
# print(medical_df.charges.corr(medical_df.children))
#
# smoker_values  = {"no":0,"yes":1}
# smoker_numeric = medical_df.smoker.map(smoker_values)
# # print(smoker_numeric)
#
#
# print(medical_df.charges.corr(smoker_numeric))
#
non_smoker_df = medical_df[medical_df.smoker == "no"]
# sns.scatterplot(data=non_smoker_df, x="age", y="charges", alpha=0.7, s=15)
# plt.title("Age vs Charges")
# plt.show()      # 显示图表
#
# def estimate_charges(age,w,b):
#     return w*age+b
# w=50
# b=100
# ages = non_smoker_df.age
#
# print(estimate_charges(40,w,b))
#
# target = non_smoker_df.charges#调用非抽烟者的年龄
# plt.plot(ages,estimate_charges(ages,w,b),"r",alpha = 0.9);#绘制折线图，r为红颜色
# plt.scatter(ages,target,s=8,alpha= 0.8);#绘制散点图，s为点的大小
# plt.xlabel("Age")
# plt.ylabel("Charges")
# plt.legend(["Estimate","Actual"])#图例
# plt.show()


