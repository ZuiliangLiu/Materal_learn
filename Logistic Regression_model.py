#用于处理分类问题
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)

    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))

    return preds




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
# print(train_probs)
# print(model.classes_)
#
# print(accuracy_score(train_targets, train_pred))
# print(accuracy_score(val_targets, val_pred))
# print(accuracy_score(test_targets, test_pred))

cm = confusion_matrix(train_targets, train_pred, normalize="true")
# print(cm)

# train_pred = predict_and_plot(X_train, train_targets, "Training")
# plt.show()
#
# val_pred = predict_and_plot(X_val, val_targets, "Validation")
# plt.show()
#
# test_pred = predict_and_plot(X_test, test_targets, "Testing")
# plt.show()

new_input = {
    'Date': '2021-06-19',
    'Location': 'Katherine',
    'MinTemp': 23.2,
    'MaxTemp': 33.2,
    'Rainfall': 10.2,
    'Evaporation': 4.2,
    'Sunshine': np.nan,
    'WindGustDir': 'NNW',
    'WindGustSpeed': 52.0,
    'WindDir9am': 'NW',
    'WindDir3pm': 'NNE',
    'WindSpeed9am': 13.0,
    'WindSpeed3pm': 20.0,
    'Humidity9am': 89.0,
    'Humidity3pm': 58.0,
    'Pressure9am': 1004.8,
    'Pressure3pm': 1001.5,
    'Cloud9am': 8.0,
    'Cloud3pm': 5.0,
    'Temp9am': 25.7,
    'Temp3pm': 33.0,
    'RainToday': 'Yes'
}#创建新的输入
new_input_df = pd.DataFrame([new_input])

new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])
X_new_input = new_input_df[numeric_cols + encoded_cols]

# print(model.predict(X_new_input))
# print(model.predict_proba(X_new_input))

aussie_rain = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}
joblib.dump(aussie_rain, 'aussie_rain.joblib')

aussie_rain2 = joblib.load('aussie_rain.joblib')

# print(aussie_rain2["model"].coef_)

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])

    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])

    X_input = input_df[numeric_cols + encoded_cols]

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]

    return pred, prob

print(predict_input(new_input))

