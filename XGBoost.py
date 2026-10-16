import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold




ross_df = pd.read_csv('train.csv', low_memory=False)
store_df = pd.read_csv('store.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

# print(ross_df.head())

merged_df = ross_df.merge(store_df, how='left', on='Store')#整合数据集和商店的特征

merged_test_df = test_df.merge(store_df, how='left', on='Store')

# print(merged_df.head())

def split_date(df):#抽取日期为特征列
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week

split_date(merged_df)
split_date(merged_test_df)

# print(merged_df.head())

# print(merged_df[merged_df.Open == 0].Sales.value_counts())

merged_df = merged_df[merged_df.Open == 1].copy()

def comp_months(df):#增加特征列：竞争门店存在的月数
    df['CompetitionOpen'] = (
        12 * (df.Year - df.CompetitionOpenSinceYear)
        + (df.Month - df.CompetitionOpenSinceMonth)
    )

    df['CompetitionOpen'] = (
        df['CompetitionOpen']
        .map(lambda x: 0 if x < 0 else x)
        .fillna(0)
    )


comp_months(merged_df)
comp_months(merged_test_df)
#特征工程（查看当前月份是否属于促销月份，促销活动开始了多久）
def check_promo_month(row):
    month2str = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    try:
        months = (row['PromoInterval'] or '').split(',')

        if row['Promo2Open'] and month2str[row['Month']] in months:
            return 1
        else:
            return 0

    except Exception:
        return 0


def promo_cols(df):
    # 计算 Promo2 开始至今的近似月数
    df['Promo2Open'] = (
        12 * (df.Year - df.Promo2SinceYear)
        + (df.WeekOfYear - df.Promo2SinceWeek) * 7 / 30
    )

    df['Promo2Open'] = (
        df['Promo2Open']
        .map(lambda x: 0 if x < 0 else x)
        .fillna(0)
        * df['Promo2']
    )

    # 判断当前月份是否属于 Promo2 的促销启动月份
    df['IsPromo2Month'] = (
        df.apply(check_promo_month, axis=1) * df['Promo2']
    )

promo_cols(merged_df)
promo_cols(merged_test_df)

input_cols = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen',
    'Day', 'Month', 'Year', 'WeekOfYear', 'Promo2',
    'Promo2Open', 'IsPromo2Month'
]

target_col = 'Sales'

inputs = merged_df[input_cols].copy()
targets = merged_df[target_col].copy()

test_inputs = merged_test_df[input_cols].copy()

#将整个数据集分为数值列与分类列
numeric_cols = [
    'Store', 'Promo', 'SchoolHoliday',
    'CompetitionDistance', 'CompetitionOpen',
    'Promo2', 'Promo2Open', 'IsPromo2Month',
    'Day', 'Month', 'Year', 'WeekOfYear'
]

categorical_cols = [
    'DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment'
]
#将数值列缺失的数据填补上
max_distance = inputs.CompetitionDistance.max()

inputs['CompetitionDistance'] = (
    inputs['CompetitionDistance'].fillna(max_distance * 2)
)

test_inputs['CompetitionDistance'] = (
    test_inputs['CompetitionDistance'].fillna(max_distance * 2)
)
# inputs['CompetitionDistance'].fillna(max_distance*2, inplace=True)
# test_inputs['CompetitionDistance'].fillna(max_distance*2, inplace=True)
#将数值进行归一化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(inputs[numeric_cols])

inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
#对分类列进行独特编码
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
).fit(inputs[categorical_cols])

encoded_cols = list(
    encoder.get_feature_names_out(categorical_cols)
)

inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X = inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# 创建模型
model = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=20,
    max_depth=4
)

model.fit(X,targets)
preds = model.predict(X)
# print(preds)

def rmse(a, b):
    return root_mean_squared_error(a, b)

print(rmse(targets, preds))

# # 线性回归：训练、预测、计算RMSE
# linear_model = LinearRegression()
# linear_model.fit(X, targets)
#
# linear_preds = linear_model.predict(X)
# linear_rmse = root_mean_squared_error(targets, linear_preds)
#
# print(f"线性回归训练集 RMSE：{linear_rmse:.2f}")
#
#
# # 随机森林：训练、预测、计算RMSE
# forest_model = RandomForestRegressor(
#     n_estimators=20,
#     max_depth=10,
#     random_state=42,
#     n_jobs=-1
# )
# forest_model.fit(X, targets)
#
# forest_preds = forest_model.predict(X)
# forest_rmse = root_mean_squared_error(targets, forest_preds)
#
# print(f"随机森林训练集 RMSE：{forest_rmse:.2f}")

def train_and_evaluate(
    X_train, train_targets,
    X_val, val_targets,
    **params
):
    model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        **params
    )

    # 使用训练集训练模型
    model.fit(X_train, train_targets)

    # 计算训练集 RMSE
    train_rmse = rmse(
        train_targets,
        model.predict(X_train)
    )

    # 计算验证集 RMSE
    val_rmse = rmse(
        val_targets,
        model.predict(X_val)
    )

    return model, train_rmse, val_rmse