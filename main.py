from scipy.io import arff
import pandas as pd
import numpy as np


def task1():
    data = arff.loadarff('messidor_features.arff')
    df = pd.DataFrame(data[0])
    df = df.replace({'?' : np.nan}).dropna()
    print(df)
    # basic statistics df
    print(df.describe(include='all'))
    # Document messidor features.arff above
    # Document hepatitis below
    df2 = pd.read_csv('hepatitis1.csv', 
    names=["class", "age", "sex", "steroid", "antivirals", 
    "fatigue", "malaise", "anorexia", "liver_big", "liver_firm", 
    "spleen_palpable", "spiders", "ascites", "varices", "bilirubin", 
    "alk_phosphate", "sgot", "albumin", "protime", "histology"])
    df2 = df2.replace({'?': np.nan}).dropna()
    print(df2)
    # basic statistics df2
    print(df2.describe(include='all'))
    df = df.to_numpy()
    df2 = df2.to_numpy()
    return df, df2 # messidor and hep data respectively

def task2(k, depth, x_train, y_train, x_test):
    import KNN
    model_knn = KNN(k)
    y_prob_knn, knn = model_knn.fit(x_train, y_train).predict(x_test)
    y_pred = np.argmax(y_prob_knn, axis = -1)
    accuracy_knn = model_knn.evaluate_acc(y_train, y_pred)
    print("Accuracy of knn", accuracy_knn)





# Diabetic Retinopathy Debrecen Data Set Data Set
if __name__ == '__main__':
    mess_data, hep_data = task1()
    
    mess_data = mess_data.astype(np.float64)
    hep_data = hep_data.astype(np.float64)
    print(hep_data)
    
    # 66% of train data and 33% of test data
    # n_mess_train, n_hep_train = mess_data.shape[0] * 2/3, hep_data.shape[0] * 2/3
    
    # mess_x_train = mess_data[:n_mess_train, :mess_data.shape[1]] # All columns except last one
    # mess_y_train = mess_data[:n_mess_train, -1] # only get last column i.e. class
    # print(mess_x_train.shape)
    # print(mess_y_train.shape)

    # mess_x_test = mess_data[n_mess_train:, :mess_data.shape[1]]
    # mess_y_test = mess_data[n_mess_train:, -1]
    # print(mess_x_test.shape)
    # print(mess_y_test.shape)

