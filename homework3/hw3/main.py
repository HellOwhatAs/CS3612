import os
import numpy as np
from dataset import get_data,get_HOG,standardize

from matplotlib import pyplot as plt

if __name__ == '__main__':
######################## Get train/test dataset ########################
    X_train,X_test,Y_train,Y_test = get_data('dataset')
########################## Get HoG featues #############################
    H_train,H_test = get_HOG(X_train), get_HOG(X_test)
######################## standardize the HoG features ####################
    H_train,H_test = standardize(H_train), standardize(H_test)
########################################################################
######################## Implement you code here #######################
########################################################################

    from sklearn.svm import SVC
    
    linearSVM, rbfSVM, polySVM = 1, 1, 1
    
    if linearSVM:
        ## Linear SVM
        ### Training
        clf = SVC(kernel="linear")
        clf.fit(H_train, Y_train)
    
        ### Accuracy
        print(f"Accuracy of Linear SVM: {np.sum(clf.predict(H_test) == Y_test) / Y_test.shape[0]}")
    
        ### Count support vectors
        print(f"Number of support vectors: {np.sum(clf.n_support_)}")
    
        ### Count positive and negative support vectors
        print("Number of negative support vectors: {}\nNumber of positive support vectors: {}".format(*clf.n_support_))
    
        ### Visualize the top 20 images
        idx = np.argsort(clf.dual_coef_[0])
        def find_original_idx(idx_in_support_vector: int) -> int:
            '''
            return the original index of support_vectors_[idx_in_support_vector] in H_train.
            '''
            return np.where(np.equal(H_train, clf.support_vectors_[idx_in_support_vector]).all(axis=1))[0][0]
    
        fig, ax = plt.subplots(4, 5, figsize=(15, 10))
        fig.tight_layout(h_pad=1)
        for i, elem in enumerate(idx[:20]):
            plt.subplot(4, 5, i + 1)
            ori_idx = find_original_idx(elem)
            plt.imshow(X_train[ori_idx], cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.title(r"$\alpha_{" + str(ori_idx) + r"}$ = " + str(clf.dual_coef_[0][elem] / np.sign(Y_train[ori_idx] - 0.5)))
        plt.savefig("airplane.svg")
        plt.show()
    
        fig, ax = plt.subplots(4, 5, figsize=(15, 10))
        fig.tight_layout(h_pad=1)
        for i, elem in enumerate(idx[-20:]):
            plt.subplot(4, 5, i + 1)
            ori_idx = find_original_idx(elem)
            plt.imshow(X_train[ori_idx], cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.title(r"$\alpha_{" + str(ori_idx) + r"}$ = " + str(clf.dual_coef_[0][elem] / np.sign(Y_train[ori_idx] - 0.5)))
        plt.savefig("bird.svg")
        plt.show()
    
    if rbfSVM:
        ## RBF kernel SVM
        ### Training & Testing with default hyper-parameters
        clf_rbf = SVC(kernel='rbf')
        clf_rbf.fit(H_train, Y_train)
        print(f"Accuracy of RBF kernel SVM with default hyper-parameters: {np.sum(clf_rbf.predict(H_test) == Y_test) / Y_test.shape[0]}")
    
        ### Training & Testing with tuned gamma
        clf_rbf = SVC(kernel='rbf', gamma = 0.005)
        clf_rbf.fit(H_train, Y_train)
        print(f"Accuracy of RBF kernel SVM with gamma = 0.005: {np.sum(clf_rbf.predict(H_test) == Y_test) / Y_test.shape[0]}")
    
    if polySVM:
        ## Polynomial kernel SVM
        ### Training & Testing with default hyper-parameters
        clf_poly = SVC(kernel='poly')
        clf_poly.fit(H_train, Y_train)
        print(f"Accuracy of Polynomial kernel SVM with default hyper-parameters: {np.sum(clf_poly.predict(H_test) == Y_test) / Y_test.shape[0]}")
    
        ### Training & Testing with tuned hyper-parameters
        clf_poly = SVC(kernel='poly', coef0 = 0.25, degree = 3)
        clf_poly.fit(H_train, Y_train)
        print(f"Accuracy of Polynomial kernel SVM with coef0 = 0.25, degree = 3: {np.sum(clf_poly.predict(H_test) == Y_test) / Y_test.shape[0]}")