import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from medical_program import non_smoker_df, medical_df

model = LinearRegression()
inputs = non_smoker_df[["age"]]  # 输入特征：年龄，保留二维表格
targets = non_smoker_df["charges"]  # 预测目标：费用

def rmse(targets, predictions):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    errors = targets - predictions   # 真实值与预测值的差
    mse = np.mean(errors ** 2)        # 平方后求平均
    return np.sqrt(mse)               # 开平方，得到 RMSE
# print("inputs.shape:", inputs.shape)
# print("targets.shape:", targets.shape)

model.fit(inputs, targets)
LinearRegression()
# print(model.predict(np.array([[23],[52],[65]])))

def estimate_charges(age, w, b):
    return w * age + b

def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges

    estimated_charges = estimate_charges(ages, w, b)

    plt.plot(ages, estimated_charges, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual'])
    plt.show()

prediction = model.predict(np.array(inputs))
# loss = rmse(targets, prediction)
# print(loss)
# print(prediction)
# print(model.coef_)#打印W
# print(model.intercept_)#打印b

# try_parameters(model.coef_, model.intercept_)
#
# inputs_2 = non_smoker_df[["age","bmi"]]
# targets_2 = non_smoker_df.charges
# model2 = LinearRegression().fit(inputs_2, targets_2)
# predictions_2 = model2.predict(inputs_2)
# loss_2 = rmse(targets_2, predictions_2)
# print(loss_2)
#在图标中添加新的编码列
# sns.barplot(data=medical_df,x="smoker",y="charges")
# plt.show()
smoker_codes={"no":0,"yes":1}
medical_df["smoker_codes"] = medical_df.smoker.map(smoker_codes)
# print(medical_df.charges.corr(medical_df.smoker_codes))
# inputs_3 = medical_df[["age","bmi","children","smoker_codes"]]
# targets_3 = medical_df.charges
# model3 = LinearRegression().fit(inputs_3, targets_3)
# predictions_3 = model3.predict(inputs_3)
# loss_3 = rmse(targets_3, predictions_3)
# print(loss_3)
#
# sns.barplot(data=medical_df,x="sex",y="charges")
# plt.show()
sex_codes={"female":0,"male":1}
medical_df["sex_codes"] = medical_df.sex.map(sex_codes)
# print(medical_df.charges.corr(medical_df.sex_codes))
#
# inputs_4 = medical_df[["age","bmi","children","smoker_codes","sex_codes"]]
# targets_4 = medical_df.charges
# model4 = LinearRegression().fit(inputs_4, targets_4)
# predictions_4 = model4.predict(inputs_4)
# loss_4 = rmse(targets_4, predictions_4)
# print(loss_4)

enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])#识别地区类别
# print(enc.categories_)#保存识别的类型

one_hot = enc.transform(medical_df[['region']]).toarray()#将地区转换为独热编码数组；.toarray()：把默认返回的稀疏矩阵转换成普通 NumPy 数组。
# print(one_hot)
medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot#添加编码到数据集
# print(medical_df)

inputs_5 = medical_df[["age","bmi","children","smoker_codes","sex_codes","northeast","northwest","southeast","southwest"]]
targets_5 = medical_df.charges
model5 = LinearRegression().fit(inputs_5, targets_5)
predictions_5 = model5.predict(inputs_5)
loss_5 = rmse(targets_5, predictions_5)
# print(loss_5)

# print(inputs_5.loc[10])
prediction=model5.predict([[25,26,0,0,1,1,0,0,0]])
print(prediction)
print(model5.coef_)

numeric_cols = ['age', 'bmi', 'children']

scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])

scaled_inputs = scaler.transform(medical_df[numeric_cols])
# print(scaled_inputs)

# 提取已经编码的分类特征
cat_cols = [
    'smoker_codes', 'sex_codes',
    'northeast', 'northwest', 'southeast', 'southwest'
]
categorical_data = medical_df[cat_cols].values

# 按列拼接两部分特征
inputs_final = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets_final = medical_df.charges
model_final = LinearRegression().fit(inputs_final, targets_final)
predictions_final = model_final.predict(inputs_final)
loss_final= rmse(targets_final, predictions_final)
# print(loss_final)
print(model_final.coef_)