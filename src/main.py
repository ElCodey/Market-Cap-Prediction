import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def preprocessing(df):
    
    df = df.drop(["company", "rank", "rank_change", "newcomer", "prev_rank", "CEO", "Website", "Ticker"], axis=1)
    
    df["ceo_founder"] = df["ceo_founder"].replace({"no": 0, "yes": 1})
    df["ceo_woman"] = df["ceo_woman"].replace({"no": 0, "yes": 1})
    df["profitable"] = df["profitable"].replace({"no": 0, "yes": 1})
    
    #Market cap null valuess coded as "-"
    df["Market Cap"] = df["Market Cap"].replace("-", np.NaN)
    df["Market Cap"] = df["Market Cap"].astype(float)
    df = df.dropna(subset=["Market Cap"])
    
    profit_col_mean = df["profit"].mean()
    df["profit"] = df["profit"].fillna(profit_col_mean)
    
    dummy_cols = pd.get_dummies(df[["sector", "city", "state"]])
    df = pd.concat([df, dummy_cols], axis=1)
    df = df.drop(["sector", "city", "state"], axis=1)
    
    
    return df

def scale_and_split_data(df):
    X = df.drop("Market Cap", axis=1)
    y = df["Market Cap"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

def linear_models(X_train, X_test, y_train, y_test):
    linear_model = LinearRegression()
    lasso_model = Lasso()
    ridge_model = Ridge()
    
    linear_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    
    linear_y_predict = linear_model.predict(X_test)
    lasso_y_predict = lasso_model.predict(X_test)
    ridge_y_predict = ridge_model.predict(X_test)
    
    linear_rmse = mean_squared_error(y_test, linear_y_predict)
    lasso_rmse = mean_squared_error(y_test, lasso_y_predict)
    ridge_rmse = mean_squared_error(y_test, ridge_y_predict)
    
    linear_r2 = r2_score(y_test, linear_y_predict)
    lasso_r2 = r2_score(y_test, lasso_y_predict)
    ridge_r2 = r2_score(y_test, ridge_y_predict)
    
    return [[linear_rmse, linear_r2], [lasso_rmse, lasso_r2], [ridge_rmse, ridge_r2]]

def ensemble_models(X_train, X_test, y_train, y_test):
    gradient_boost_model = GradientBoostingRegressor()
    random_forest_model = RandomForestRegressor()
    
    gradient_boost_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    
    gradient_y_predict = gradient_boost_model.predict(X_test)
    random_forest_y_predict = random_forest_model.predict(X_test)
    
    gradient_rmse = mean_squared_error(y_test, gradient_y_predict)
    forest_rmse = mean_squared_error(y_test, random_forest_y_predict)
    
    gradient_r2 = r2_score(y_test, gradient_y_predict)
    forest_r2 = r2_score(y_test, random_forest_y_predict)
    
    return [[gradient_rmse, gradient_r2], [forest_rmse, forest_r2]]