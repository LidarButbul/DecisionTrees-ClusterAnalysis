import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from LogisticModelTree import LogisticModelTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import time

def analyze_data(data):
    # bar plot for target column distribution
    data['target'].value_counts().plot(kind='bar', figsize=(6,4), title="Number for each label")
    plt.xticks(rotation=0)
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.show()

    # null values per column
    print("Null values:\n" , data.isna().sum())

    # statistical table for numeric features
    print(data[["age", "trestbps", "chol", 'thalach', "oldpeak"]].describe())

    # histogram for each feature
    for col in data.columns:
        plt.hist(data[col])
        plt.title(col)
        plt.xlabel("Values")
        plt.ylabel("Count")
        plt.show()

    # another analysis - correlation analysis
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', ax=ax)
    plt.show()


def preprocess(data):
    # fill missing values
    data['age'].fillna(value=data['age'].mean(), inplace=True)
    data['sex'].fillna(data['sex'].mode()[0], inplace=True)
    data['trestbps'].fillna(value=data['trestbps'].mean(), inplace=True)
    data['chol'].fillna(value=data['chol'].mean(), inplace=True)
    data['fbs'].fillna(data['fbs'].mode()[0], inplace=True)
    data['restecg'].fillna(data['restecg'].mode()[0], inplace=True)
    data['thalach'].fillna(value=data['thalach'].mean(), inplace=True)
    data['exang'].fillna(data['exang'].mode()[0], inplace=True)
    data['oldpeak'].fillna(data['oldpeak'].median(), inplace=True)
    data['slope'].fillna(data['slope'].mode()[0], inplace=True)
    data['ca'].fillna(data['ca'].mode()[0], inplace=True)
    data['thal'].fillna(data['thal'].mode()[0], inplace=True)

    # create matrix X for the features and vector Y for the label
    X = data.iloc[:, :-1].to_numpy()
    Y = data.iloc[:, -1].to_numpy()

    # normalize the matrix X
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, Y

    # bonus
def calculate_class_weights(y):
    pass

def acc_time(x,y):
    start = time.time()
    kf = KFold(n_splits=10)
    scores = []
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticModelTree().fit(X_train, y_train)
        pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, pred))
    end = time.time()
    return np.mean(scores), end - start


if __name__ == "__main__":
    data = pd.read_csv('heart.csv')
    x, y = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=11)
    # weights = calculate_class_weights(y)

    # implement here the experiments for task 4
    # fold cv without discretization
    print("results without discretization")
    print("mean accuracy score:", acc_time(x, y)[0])
    print("running time:", acc_time(x, y)[1])

    # chosen features for discretization
    features = ["age", "trestbps", "chol", 'thalach', "oldpeak"]

    # fold cv with equal width discretization with 5 bins
    df1 = pd.DataFrame(x, columns=data.columns[:-1])
    for feat in features:
        bins = pd.cut(df1[feat], bins=5)
        bin_means = bins.apply(lambda f: f.mid)
        df1[feat] = bin_means
        df1[feat] = pd.to_numeric(df1[feat])
    x1 = df1.to_numpy()
    print('results with equal width discretization with 5 bins')
    print("mean accuracy score:", acc_time(x1, y)[0])
    print("running time:", acc_time(x1, y)[1])

    # fold cv with equal width discretization with 10 bins
    df2 = pd.DataFrame(x, columns=data.columns[:-1])
    for feat in features:
        bins = pd.cut(df2[feat], bins=10)
        bin_means = bins.apply(lambda f: f.mid)
        df2[feat] = bin_means
        df2[feat] = pd.to_numeric(df2[feat])
    x2 = df2.to_numpy()
    print('results with equal width discretization with 10 bins')
    print("mean accuracy score:", acc_time(x2, y)[0])
    print("running time:", acc_time(x2, y)[1])

    # fold cv with equal frequency discretization
    df3 = pd.DataFrame(x, columns=data.columns[:-1])
    for feat in features:
        bins = pd.qcut(df3[feat], 5, duplicates='drop')
        bin_means = bins.apply(lambda f: f.mid)
        df3[feat] = bin_means
        df3[feat] = pd.to_numeric(df3[feat])
    x3 = df3.to_numpy()
    print('results with equal frequency discretization with 5 bins')
    print("mean accuracy score:", acc_time(x3, y)[0])
    print("running time:", acc_time(x3, y)[1])

    # results plots
    acc = [acc_time(x, y)[0], acc_time(x1, y)[0], acc_time(x2, y)[0], acc_time(x3, y)[0]]
    run_time = [acc_time(x, y)[1], acc_time(x1, y)[1], acc_time(x2, y)[1], acc_time(x3, y)[1]]
    labels = ['no discretization', 'equal width - 5', 'equal width - 10', 'equal freq - 5']
    plt.bar(labels, acc)
    plt.title('Accuracy Comparison')
    plt.ylim(0.5,0.8)
    plt.show()
    plt.bar(labels, run_time)
    plt.title('Running Time Comparison')
    plt.show()


