import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
import numpy as np


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    sum_i = 0
    sum_j = 0
    for i in range(len(C)):
        for j in range(len(C[i])):
            sum_j = sum_j + C[i][j]
            if i == j:
                sum_i = sum_i + C[i][j]
    return sum_i / sum_j


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall = [0] * len(C)
    for k in range(len(C)):
        sum_j = 0
        for j in range(len(C[k])):
            sum_j = sum_j + C[k][j]
        recall[k] = C[k][k] / sum_j
    return recall


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision = [0] * len(C)
    for i in range(len(C)):
        for j in range(len(C[i])):
            precision[i] = precision[i] + C[i][j]
    for k in range(len(C)):
        precision[k] = C[k][k] / precision[k]
    return precision
            

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    iBest = 1
    max_accuracy = 0
    classifiers = [SGDClassifier, GaussianNB, RandomForestClassifier, MLPClassifier, AdaBoostClassifier]
    clf_names = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']    
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i in range(len(classifiers)):
            clf = classifiers[i]()
            clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            confusion = confusion_matrix(y_test, predicted)
            accur = accuracy(confusion)
            outf.write(f'Results for {clf_names[i]}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {accur:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall(confusion)]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision(confusion)]}\n')
            outf.write(f'\tConfusion Matrix: \n{confusion}\n\n')
            if accur > max_accuracy:
                max_accuracy = accur
                iBest = i

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2Format the topk= 5 feature indices extracted from the 32K training set to file using the formatstring provided
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    classifiers = [SGDClassifier, GaussianNB, RandomForestClassifier, MLPClassifier, AdaBoostClassifier]
    clf = classifiers[iBest]()
    
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:      
        # For each number of training examples, compute results and write the following output:
        for num_train in [1000, 5000, 10000, 15000, 20000]:
            clf.fit(X_train[:num_train], y_train[:num_train])
            predicted = clf.predict(X_test)
            confusion = confusion_matrix(y_test, predicted)
            outf.write(f'{num_train}: {accuracy(confusion):.4f}\n')

    return X_train[:1000], y_train[:1000]


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    classifiers = [SGDClassifier, GaussianNB, RandomForestClassifier, MLPClassifier, AdaBoostClassifier]
    clf = classifiers[i]()    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # for each number of features k_feat, write the p-values for that number of features:
        for k_feat in [5, 50]:
            selector = SelectKBest(f_classif, k_feat)
            X_new = selector.fit_transform(X_train, y_train)
            p_values = selector.pvalues_ 
            outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        
        # Train the best classifier from section 3.1 for each of the 1K training set and the 32K training set,
        # using only the best k= 5 features.
        k = 5
        
        selector_1k = SelectKBest(f_classif, k)
        X_new = selector_1k.fit_transform(X_1k, y_1k)
        X_test_new = selector_1k.transform(X_test)
        clf.fit(X_new, y_1k)
        predicted_1k = clf.predict(X_test_new)
        confusion_1k = confusion_matrix(y_test, predicted_1k)
        accuracy_1k = accuracy(confusion_1k)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        
        selector_full = SelectKBest(f_classif, k)
        X_new = selector_full.fit_transform(X_train, y_train)
        X_test_new = selector_full.transform(X_test)
        clf.fit(X_new, y_train)
        predicted_full = clf.predict(X_test_new)
        confusion_full = confusion_matrix(y_test, predicted_full)  
        accuracy_full = accuracy(confusion_full)
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        
        # Extract the indices of the topk= 5 features using the 1K training set and take the intersection withthe
        # k= 5 features using the 32K training set.
        top_5_1k = sorted(range(len(selector_1k.pvalues_ )), key = lambda sub: selector_1k.pvalues_ [sub])[-k:]
        top_5_full = sorted(range(len(selector_full.pvalues_ )), key = lambda sub: selector_full.pvalues_ [sub])[-k:]
        feature_intersection = list(set(top_5_1k) & set(top_5_full)) 
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        
        # Format the topk= 5 feature indices extracted from the 32K training set to file using the formatstring
        # provided
        outf.write(f'Top-5 at higher: {top_5_full}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
    '''
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    kf = KFold(n_splits=5, shuffle=True)
    accuracies = []
    classifiers = [SGDClassifier, GaussianNB, RandomForestClassifier, MLPClassifier, AdaBoostClassifier]      
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]    
            kfold_accuracies = []
            for j in range(len(classifiers)):
                clf = classifiers[j]()
                clf.fit(X_train, y_train)
                predicted = clf.predict(X_test)
                confusion = confusion_matrix(y_test, predicted)
                kfold_accuracies.append(accuracy(confusion))
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            accuracies.append(kfold_accuracies)

        p_values = []
        accuracies = np.array(accuracies)
        i_list = accuracies[:, i]
        for a in range(0, 5):
            if a == i:
                continue
            a_list = accuracies[:, a]
            S = ttest_rel(a_list.tolist(), i_list.tolist())
            p_values.append(S.pvalue)
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    output_dir = args.output_dir
    # load data and split into train and test.
    loaded = np.load(args.input)['arr_0']
    X_train, X_test, y_train, y_test = train_test_split(loaded[:, :173], loaded[:, 173], test_size = .20, shuffle=True)
    # complete each classification experiment, in sequence.
    iBest = class31(output_dir, X_train, X_test, y_train, y_test)
    (X_1k, y_1k) = class32(output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(output_dir, X_train, X_test, y_train, y_test, iBest)
