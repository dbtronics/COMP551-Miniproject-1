from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def task1():
    data = arff.loadarff('messidor_features.arff')
    df = pd.DataFrame(data[0])
    df = df.replace({'?' : np.nan}).dropna()
    # print(df)
    # basic statistics df
    # print(df.describe(include='all'))
    # Document messidor features.arff above
    # Document hepatitis below
    df2 = pd.read_csv('hepatitis1.csv', 
    names=["class", "age", "sex", "steroid", "antivirals", 
    "fatigue", "malaise", "anorexia", "liver_big", "liver_firm", 
    "spleen_palpable", "spiders", "ascites", "varices", "bilirubin", 
    "alk_phosphate", "sgot", "albumin", "protime", "histology"])
    df2 = df2.replace({'?': np.nan}).dropna()
    # print(df2)
    # basic statistics df2
    # print(df2.describe(include='all'))
    df = df.to_numpy()
    df2 = df2.to_numpy()
    return df, df2 # messidor and hep data respectively

def task2(k, depth, x_train, y_train, x_test):
    import KNN as kmodel
    model_knn = kmodel.KNN(k)
    y_prob_knn, knn = model_knn.fit(x_train, y_train).predict(x_test)
    y_pred = np.argmax(y_prob_knn, axis = -1)
    accuracy_knn = model_knn.evaluate_acc(y_train, y_pred)
    return accuracy_knn

def decision_boundary(x, x_train, y_train):
    #we can make the grid finer by increasing the number of samples from 200 to higher value
    x0v = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 200)
    x1v = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 200)

    #to features values as a mesh  
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(),x1.ravel())).T

    for k in range(1,4):
        import KNN as kmodel
        model = kmodel.KNN(K=k)

        C = np.max(y_train) + 1
        y_train_prob = np.zeros((y_train.shape[0], C))
        y_train_prob[np.arange(y_train.shape[0]), y_train] = 1

        #to get class probability of all the points in the 2D grid
        y_prob_all, _ = model.fit(x_train, y_train).predict(x_all)

        y_pred_all = np.zeros_like(y_prob_all)
        y_pred_all[np.arange(x_all.shape[0]), np.argmax(y_prob_all, axis=-1)] = 1
        print(x_train[:,0].shape)
        print(x_train[:,1].shape)
        print(y_train_prob.shape)
        plt.scatter(x_train[:,0], x_train[:,1], c=y_train_prob, marker='o', alpha=1)
        plt.scatter(x_all[:,0], x_all[:,1], c=y_pred_all, marker='.', alpha=0.01)
        plt.show()




# Diabetic Retinopathy Debrecen Data Set Data Set
if __name__ == '__main__':
    mess_data, hep_data = task1()
    
    mess_data = mess_data.astype(np.float64)
    hep_data = hep_data.astype(np.float64)

    # parameter adjustments
    k_range = 6
    train_portion = 8/10 # meaning 1/3 will be test portion
    num_features = 2
    
    # 66% of train data and 33% of test data
    n_mess_train, n_hep_train = int(mess_data.shape[0] * train_portion), int(hep_data.shape[0] * train_portion)
    
    # train data set for messidore
    mess_x_train = mess_data[:n_mess_train, :num_features] # All columns except last one
    # mess_y_train = mess_data[:n_mess_train, -1].reshape(n_mess_train, 1) # only get last column i.e. class
    mess_y_train = mess_data[:n_mess_train, -1]
    mess_y_train = mess_y_train.astype(np.int32)
    # test data set for messidore
    mess_x_test = mess_data[n_mess_train:, :num_features]
    # mess_y_test = mess_data[n_mess_train:, -1].reshape(mess_data.shape[0] - n_mess_train, 1)
    mess_y_test = mess_data[n_mess_train:, -1]
    mess_y_test = mess_y_test.astype(np.int32)

    #train data set for hepatitis
    hep_x_train = hep_data[:n_hep_train, 1:1+num_features]
    # hep_y_train = hep_data[:n_hep_train, 0].reshape(n_hep_train, 1)
    hep_y_train = hep_data[:n_hep_train, 0]
    hep_y_train = hep_y_train.astype(np.int32)
    #test data set for hepatitis
    hep_x_test = hep_data[n_hep_train:, 1:1+num_features]
    # hep_y_test = hep_data[n_hep_train:, 0].reshape(hep_data.shape[0] - n_hep_train, 1)
    hep_y_test = hep_data[n_hep_train:, 0]
    hep_y_test = hep_y_test.astype(np.int32)

    ## We have test and train for both
    ## Let's work from here
    for i in range(1, k_range):
        print("K = ", i)
        print("Messidore Accuracy: ", task2(i, 9, mess_x_train, mess_y_train, mess_x_test))
        print("Hepatitis Accuracy: ", task2(i, 9, hep_x_train, hep_y_train, hep_x_test))
        print("\n")

    decision_boundary(mess_data, mess_x_train, mess_y_train)