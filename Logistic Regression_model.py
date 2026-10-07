#用于处理分类问题
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score


raw_df = pd.read_csv("weatherAUS.csv")
# print(raw_df.head())
# print(raw_df.info())

raw_df.dropna(subset=["RainToday","RainTomorrow"],inplace=True)
# print(raw_df.info())

# fig1 = px.histogram(raw_df,x="Location",title="Location vs. Rainy Days",color="RainToday")
# fig1.show()

# fig2 = px.histogram(raw_df,x="Temp3pm",title="Temperature at 3 pm vs.Rain Tomorrow",color="RainTomorrow")
# fig2.show()

# fig3 = px.histogram(raw_df,x="RainTomorrow",title="Rain Today vs.Rain Tomorrow",color="RainToday")
# fig3.show()

# fig4 = px.scatter(raw_df,x="MinTemp",y="MaxTemp",title="MinTemp vs. MaxTemp",color="RainToday")
# fig4.show()

# fig5 = px.strip(raw_df.sample(5000),x="Temp3pm",y="Humidity3pm",title="Temp(3pm) vs. Humidity",color="RainTomorrow")
# fig5.show()#绘图

use_sample = False #不执行抽样
sample_fraction = 0.1
if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()#抽样开关

# train_val_df,test_df = train_test_split(raw_df,test_size=0.2,random_state=42)#20%为测试集
# train_df,val_df = train_test_split(train_val_df,test_size=0.25,random_state=42)#20%为验证集，其余为训练集#一般情况下创建训练集、测试集、验证集
# print("train_df.shape:",train_df.shape)
# print("val_df.shape:",val_df.shape)
# print("test_df.shape:",test_df.shape)

# plt.title("No. of Rows per Year")
# sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
# plt.show()
#设置训练集、测试集、验证集
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year<2015]
val_df = raw_df[year==2015]
test_df = raw_df[year>2015]
# print("train_df.shape:",train_df.shape)
# print("val_df.shape:",val_df.shape)
# print("test_df.shape:",test_df.shape)

input_cols = list(train_df.columns)[1:-1]#获取输入特征，第二列至倒数第二列
target_col = "RainTomorrow"

#为三个数据集设置特征与标签
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()#数值列
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()#分类列

# print(train_inputs[numeric_cols].describe())
# print(train_inputs[categorical_cols].nunique())

imputer = SimpleImputer(strategy="mean")#SimpleImputer：scikit-learn 提供的填补工具。
# print(raw_df[numeric_cols].isna().sum())#检查有多少缺失值

imputer.fit(raw_df[numeric_cols])#fit() 在这里表示从数据中学习填补所需的均值
# print(numeric_cols)
# print(list(imputer.statistics_))#statistics_ 是保存填补规则的地方
#替换三个数据集中的缺失值
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# print(train_inputs[numeric_cols])#查看替换后的训练集的数值列
# print(train_inputs[numeric_cols].isna().sum())#再次检查有无缺失值

scaler = MinMaxScaler()#最小最大标量器
scaler.fit(raw_df[numeric_cols])
#将数据集中数值列中的数据等比例缩放在0-1之间
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# print(train_inputs[numeric_cols].describe())

encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')#sparse_output=False：转换结果输出为普通的 NumPy 数组。
#handle_unknown='ignore'：转换时，如果遇到学习阶段没见过的类别，不报错，该特征对应的编码列全部填 0
encoder.fit(raw_df[categorical_cols])#提取所有类别型列
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))#保存这些列名
#分类列中的值转换成独热编码，然后把编码结果添加到新的列中
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# print(train_inputs[encoded_cols].head())
# pd.set_option('display.max_columns', None)
# print(train_inputs)
# train_inputs.to_csv('train_inputs.csv',index=False,encoding='utf-8-sig')
#将三个数据集进行本地保存
# train_inputs.to_parquet('train_inputs.parquet')
# val_inputs.to_parquet('val_inputs.parquet')
# test_inputs.to_parquet('test_inputs.parquet')
#
# pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
# pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
# pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

model = LogisticRegression(solver="liblinear")
model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)

# print(numeric_cols + encoded_cols)
# print(model.coef_.tolist())
# print(model.intercept_)

weights_df = pd.DataFrame({
    'feature': numeric_cols + encoded_cols,
    'weight': model.coef_.tolist()[0]
})
# print(weights_df)

# plt.figure(figsize=(10, 30))
# sns.barplot( data=weights_df.sort_values('weight', ascending=False).head(10), x='weight', y='feature')
# plt.show()
#
# plt.figure(figsize=(10, 30))
# sns.barplot( data=weights_df.sort_values('weight', ascending=True).head(10), x='weight', y='feature')
# plt.show()

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

train_probs = model.predict_proba(X_train)
print(train_probs)
print(model.classes_)

print(accuracy_score(train_targets, train_pred))
print(accuracy_score(val_targets, val_pred))
print(accuracy_score(test_targets, test_pred))




