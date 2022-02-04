from cProfile import label
import numpy as np
#the output of plotting commands is displayed inline within frontends
# %matplotlib inline                                  
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace         #for debugging 

#it is important to set the seed for reproducibility as it initializes the random number generator
np.random.seed(1234)

#define the metric we will use to measure similarity
#if the input shapes are [1,N1,F] and [N2,1,F] then output shape is [N2,N1]
#as numpy supports broadcasting with arithmetic operations
#for more on numpy broadcasting refer to: https://numpy.org/doc/stable/user/basics.broadcasting.html   
euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)

class KNN:

    def __init__(self, K=1, dist_fn= euclidean):
        self.dist_fn = dist_fn
        self.K = K
        return
    
    def fit(self, x, y):
        ''' Store the training data using this method as it is a lazy learner'''
        self.x = x
        self.y = y
        self.C = np.max(y) + 1
        return self
    
    def predict(self, x_test):
        ''' Makes a prediction using the stored training data and the test data given as argument'''
        num_test = x_test.shape[0]
        #calculate distance between the training & test samples and returns an array of shape [num_test, num_train]
        distances = self.dist_fn(self.x[None,:,:], x_test[:,None,:])
        #ith-row of knns stores the indices of k closest training samples to the ith-test sample 
        knns = np.zeros((num_test, self.K), dtype=int)
        #ith-row of y_prob has the probability distribution over C classes
        y_prob = np.zeros((num_test, self.C))
        for i in range(num_test):
            knns[i,:] = np.argsort(distances[i])[:self.K]
            y_prob[i,:] = np.bincount(self.y[knns[i,:]], minlength=self.C) #counts the number of instances of each class in the K-closest training samples
        #y_prob /= np.sum(y_prob, axis=-1, keepdims=True)
        #simply divide by K to get a probability distribution
        y_prob /= self.K
        return y_prob, knns

    def evaluate_acc(self, y_test, y_pred):
        correct_pred = y_test = y_pred # gets all the index of correct prediction
        accuracy = np.sum(correct_pred)/y_test.shape[0] * 100 # in percentage
        return accuracy

from sklearn import datasets
#to read more about load_iris() function refer to: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
dataset = datasets.load_iris()
#uncomment this if you want to see the description of the dataset 
#print(dataset.DESCR)
#uncomment to see the details about the dataset class
#help(dataset)

x, y = dataset['data'][:,:2], dataset['target']                                     #slices the first two columns or features from the data

#print the feature shape and classes of dataset 
(N,D), C = x.shape, np.max(y)+1
print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')

inds = np.random.permutation(N)    #generates an indices array from 0 to N-1 and permutes it 

#split the dataset into train and test
x_train, y_train = x[inds[:100]], y[inds[:100]]
x_test, y_test = x[inds[100:]], y[inds[100:]]

#########################################################
## TESTING KNN and understand the function of the code ##
#########################################################
n_params = 11
accuracy_array = np.zeros(n_params)
for i in range (11):
    model = KNN(K=i)
    y_prob, knns = model.fit(x_train, y_train).predict(x_test)
    print('K = ', i)
    print('knns shape:', knns.shape)
    print('y_prob shape:', y_prob.shape)
    y_pred = np.argmax(y_prob, axis = -1)
    accuracy = model.evaluate_acc(y_train, y_pred)
    print('accuracy = ', accuracy)
    accuracy_array[i] = accuracy
    print('\n')

knn = np.argmax(accuracy_array)
print('Accuracy with KNN', knn)
print(accuracy_array[knn])

plt.scatter(np.arange(n_params), accuracy_array)
plt.xlabel('KNN parameter')
plt.ylabel('Accuracy in %')
plt.show()