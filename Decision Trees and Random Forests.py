import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
warnings.filterwarnings("ignore",category=pd.errors.PerformanceWarning)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree, export_text


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

raw_df = pd.read_csv("weatherAUS.csv")#读取数据集
raw_df.dropna(subset=['RainTomorrow'], inplace=True)#：删除 RainTomorrow 列中存在缺失值的整行数据

year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year<2015]
val_df = raw_df[year==2015]
test_df = raw_df[year>2015]

input_cols = list(train_df.columns)[1:-1]#获取输入特征
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

imputer = SimpleImputer(strategy="mean")

imputer.fit(raw_df[numeric_cols])
#填充缺失列
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])
#归一化
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
#独热编码
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoder.fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, train_targets)

train_preds = model.predict(X_train)

# print(train_preds)
# print(pd.Series(train_preds).value_counts())
# print(model.predict_proba(X_train))
# print("训练集准确率：", accuracy_score(train_targets, train_preds))
# print(confusion_matrix(train_targets, train_preds))
# print(model.score(X_train, train_targets))

# print(model.score(X_val, val_targets))

# plt.figure(figsize=(20, 10))
# plot_tree(model,feature_names=X_train.columns,max_depth=2,filled=True,fontsize=10)
# plt.tight_layout()
# plt.show()

# print(model.tree_.max_depth)

# tree_text = export_text(model,max_depth=10,feature_names=list(X_train.columns))
# print(tree_text[:5000])

# print(model.feature_importances_)
# importance_df = pd.DataFrame({"feature": X_train.columns,"importance": model.feature_importances_}).sort_values("importance", ascending=False)
# print(importance_df.head(10))
#
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importance")
# sns.barplot(data=importance_df.head(10),x="importance",y="feature")
# plt.tight_layout()
# plt.show()

# model = DecisionTreeClassifier(max_depth=3,random_state=42)
# model.fit(X_train, train_targets)
# print(model.score(X_train, train_targets))
# print(model.score(X_val, val_targets))
# print(model.score(X_test, test_targets))

def max_depth_error(md):
    model = DecisionTreeClassifier(max_depth=md, random_state=42)
    model.fit(X_train, train_targets)

    train_error = 1 - model.score(X_train, train_targets)
    val_error = 1 - model.score(X_val, val_targets)

    return {"Max Depth": md,"Training Error": train_error,"Validation Error": val_error}

# errors_df = pd.DataFrame([max_depth_error(md) for md in range(1, 21)])
# # print(errors_df)
#
# plt.figure()
# plt.plot(errors_df['Max Depth'],errors_df['Training Error'])
# plt.plot(errors_df['Max Depth'],errors_df['Validation Error'])
#
# plt.title('Training vs. Validation Error')
# plt.xticks(range(0, 21, 2))
# plt.xlabel('Max Depth')
# plt.ylabel('Prediction Error (1 - Accuracy)')
# plt.legend(['Training', 'Validation'])
#
# plt.tight_layout()
# plt.show()

# model = DecisionTreeClassifier(max_depth=7,random_state=42)
# model.fit(X_train, train_targets)
# print(model.score(X_train, train_targets))
# print(model.score(X_val, val_targets))

# model = DecisionTreeClassifier(max_leaf_nodes=128,random_state=42)
#
# model.fit(X_train, train_targets)
#
# print("训练集准确率：", model.score(X_train, train_targets))
# print("验证集准确率：", model.score(X_val, val_targets))
# print("决策树实际最大深度：", model.tree_.max_depth)

# results = []
#
# best_model = None
# best_params = None
# best_val_acc = -1
#
# for md in range(1, 21):
#     for mln in [16, 32, 64, 128, 256]:
#
#         # 创建并训练当前组合的模型
#         candidate = DecisionTreeClassifier(
#             max_depth=md,
#             max_leaf_nodes=mln,
#             random_state=42
#         )
#         candidate.fit(X_train, train_targets)
#
#         # 计算准确率
#         train_acc = candidate.score(X_train, train_targets)
#         val_acc = candidate.score(X_val, val_targets)
#
#         # 记录结果
#         results.append({
#             'Max Depth': md,
#             'Max Leaf Nodes': mln,
#             'Training Accuracy': train_acc,
#             'Validation Accuracy': val_acc
#         })
#
#         # 保存验证集准确率最高的模型
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model = candidate
#             best_params = {
#                 'max_depth': md,
#                 'max_leaf_nodes': mln
#             }
#
# # 按验证集准确率从高到低排列
# results_df = pd.DataFrame(results).sort_values(
#     'Validation Accuracy',
#     ascending=False
# )
#
# print(results_df.head(10))
# print("最佳参数组合：", best_params)
# print(f"最佳验证集准确率：{best_val_acc:.2%}")

# model = RandomForestClassifier(
#     n_jobs=-1,
#     random_state=42
# )
# 
# model.fit(X_train, train_targets)

# print("训练集准确率：", model.score(X_train, train_targets))
# print("验证集准确率：", model.score(X_val, val_targets))

# train_probs = model.predict_proba(X_train)
# print(train_probs)


# print(len(model.estimators_))
#
# print(model.estimators_[0])
#
# plt.figure(figsize=(20, 10))
# plot_tree(model.estimators_[0],max_depth=2,feature_names=X_train.columns,filled=True,rounded=True,fontsize=10)
# plt.tight_layout()
#
# plt.figure(figsize=(20, 10))
# plot_tree( model.estimators_[20],max_depth=2,feature_names=X_train.columns,filled=True,rounded=True,fontsize=10)
# plt.tight_layout()
# plt.show()

# importance_df = pd.DataFrame({"feature": X_train.columns,"importance": model.feature_importances_}).sort_values("importance", ascending=False)
#
# print(importance_df.head(10))
#
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importance")
# sns.barplot(data=importance_df.head(10),x="importance",y="feature")
# plt.tight_layout()
# plt.show()

# # 创建并训练基准模型
# base_model = RandomForestClassifier(random_state=42,n_jobs=-1).fit(X_train, train_targets)
# # 计算训练集、验证集准确率
# base_train_acc = base_model.score(X_train, train_targets)
# base_val_acc = base_model.score(X_val, val_targets)
#
# base_accs = base_train_acc, base_val_acc

# print(base_accs)

# model_1 = RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=10)
# model_1.fit(X_train, train_targets)
#
# model_2 = RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=500)
# model_2.fit(X_train, train_targets)
#
# print("当前模型1（训练集、验证集）：",(model_1.score(X_train, train_targets),model_1.score(X_val, val_targets)))
# print("当前模型2（训练集、验证集）：",(model_2.score(X_train, train_targets),model_2.score(X_val, val_targets)))
# print("基准模型（训练集、验证集）：", base_accs)

# def n_estimators(md):
#     model = RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=md)
#     model.fit(X_train, train_targets)
#
#     train_error = 1 - model.score(X_train, train_targets)
#     val_error = 1 - model.score(X_val, val_targets)
#
#     return {"n_estimators": md,"Training Error": train_error,"Validation Error": val_error}
#
# errors_df = pd.DataFrame([n_estimators(md) for md in range(100, 500,10)])
# # # print(errors_df)
#
# plt.figure()
# plt.plot(errors_df['n_estimators'],errors_df['Training Error'])
# plt.plot(errors_df['n_estimators'],errors_df['Validation Error'])
#
# plt.title('Training vs. Validation Error')
# plt.xticks(range(100,500, 10))
# plt.xlabel('n_estimators')
# plt.ylabel('Prediction Error (1 - Accuracy)')
# plt.legend(['Training', 'Validation'])
#
# plt.tight_layout()
# plt.show()


















