import os
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict


""" PREPROCESSING
"""


def missing_value_handling(df, thresh=10):
    df_no_nan = df.dropna(thresh=thresh)
    df_no_nan = df_no_nan.fillna(value=df_no_nan.mean())
    return df_no_nan


def remove_multivariate_outlier(df, n_neighbors=13):
    df_temp = df
    df_float64_only = df_temp.select_dtypes(include=['float64'])
    df_temp['outliers'] = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(df_float64_only) != -1
    df_temp = df_temp[df_temp['outliers'] == True]
    return df_temp


def remove_univariate_outliers(df, num_stds=3):
    dftemp = df
    df_float64_only = list(dftemp.select_dtypes(include=['float64']).columns)
    number_outliers = 0
    for feature in df_float64_only:
        feature_column = dftemp[feature]
        std = feature_column.std()
        mean = feature_column.mean()
        three_std = std * num_stds
        inlier_low = mean - three_std if mean - three_std >= 0 else 0
        inlier_high = mean + three_std

        # Initialize the count of outliers of the feature
        cnt = 0
        # Initialize the list of values for getting the index in further steps
        list_of_values = []
        for value in feature_column:
            if value < inlier_low or value > inlier_high:
                cnt += 1
                list_of_values.append(value)
        number_outliers += len(list_of_values)
        feature_column.replace(to_replace=list_of_values, value=mean, inplace=True)
    return df


def uni_multi_variate_outlier(df, n_neighbors=13, num_stds=4.5):
    df_temp = df
    df_temp = remove_multivariate_outlier(df_temp, n_neighbors=n_neighbors)
    df_temp = remove_univariate_outliers(df_temp, num_stds=num_stds)
    return df_temp


def noise_filtering(df, noise_treshold):  # noise_treshold --> float >= 0.5
    # Splitting non-gamay wines into dataframes containing two quality tiers
    df_12 = df[(df["quality"]==1) | (df["quality"]==2)].copy()
    df_34 = df[(df["quality"]==3) | (df["quality"]==4)].copy()
    df_56 = df[(df["quality"]==5) | (df["quality"]==6)].copy()
    df_789 = df[(df["quality"]==7) | (df["quality"]==8) | (df["quality"]==9)].copy()
    df_list = [df_12, df_34, df_56, df_789]

    # Linear regression for non-gamay wines
    try:
        for idf in df_list:
            if len(idf) > 0:
                x = idf.drop(columns=["quality"])
                x = x.to_numpy()  # Converting to ndarray
                y = idf["quality"]
                rgs = LinearRegression()
                y_pred = cross_val_predict(rgs, x, y, cv=3)
                idf["predicted_quality"] = y_pred
    except Exception as exc:
        print(exc)
        return df

    df_result = pd.concat(df_list)
    # Calculating deviation from true quality
    df_result["quality_deviation"] = df_result.apply(lambda row: row["predicted_quality"] - row["quality"], axis=1)
    # Only keeping instances whose quality_deviation is smaller than the noise_treshold
    # Excluding quality tiers with few instances as to limit information loss on those
    df_return = df_result[(abs(df_result["quality_deviation"]) < noise_treshold) | ((df_result["quality"]==3) | (df_result["quality"]==9))]
    # Removing columns created in this function
    df_return = df_return.drop(columns=["predicted_quality", "quality_deviation"]).copy()

    return df_return

""" FEATURE TRANSFORMATION
"""



def split_data(df, drop_x=None, label=None, test_size=.25, random_state=30):
    X = df.drop(drop_x, axis=1)
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=test_size, random_state=random_state)
    return {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def gridsearch(train_data=None, param_dict=None, estimator_obj=None, scoring="accuracy", cv=5):
    gs = GridSearchCV(estimator=estimator_obj, scoring=scoring, param_grid=param_dict, cv=cv)
    gs.fit(train_data[0], train_data[1])
    return gs


def random_forest_classifier(data):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(data['X_train'], data['y_train'])
    y_pred1 = clf.predict(data['X_test'])
    print("Accuracy:", metrics.accuracy_score(data['y_test'], y_pred1))



