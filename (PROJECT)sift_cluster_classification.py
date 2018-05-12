from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
import cv2
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def cluster_features(img_descs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters
    
    # Concatenate all descriptors in the training set together
    all_train_descriptors = [desc for desc_list in img_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)
   
    if all_train_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    print '%i descriptors before clustering' % all_train_descriptors.shape[0]

    # Cluster descriptors to get codebook
    print 'Using clustering model %s...' % repr(cluster_model)
    print 'Clustering on training set to get codebook of %i words' % n_clusters
#
    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print 'done clustering. Using clustering model to generate BoW histograms for each image.'

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print 'done generating BoW histograms.'

    return X, cluster_model

def gen_sift_features(img_labels):
    """
    Generate SIFT features for images
    Parameters:
    -----------
    labeled_img_paths : list of lists
        Of the form [[image_path, label], ...]
    Returns:
    --------
    img_descs : list of SIFT descriptors with same indicies as labeled_img_paths
    y : list of corresponding labels
    """
    # img_keypoints = {}
    img_descs = []

    print 'generating SIFT descriptors for %i images' % len(img_labels)

    for img, label in img_labels:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(gray, None)
        img_descs.append(desc)

    print 'SIFT descriptors generated.'

    y = np.array(img_labels)[:,1]

    return img_descs, y

def img_to_vect(img, cluster_model):
    """
    Given an image path and a trained clustering model (eg KMeans),
    generates a feature vector representing that image.
    Useful for processing new images for a classifier prediction.
    """
    img = img.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    
    if desc is None: return np.array([-1]*n_clus)
    
    clustered_desc = cluster_model.predict(desc)
    img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)

    # reshape to an array containing 1 array: array[[1,2,3]]
    # to make sklearn happy (it doesn't like 1d arrays as data!)
    return img_bow_hist.reshape(1,-1)

def gridsearch(model,param_grid,scoring='f1_macro'):
    clf = GridSearchCV(model, param_grid, cv=5,
                       scoring=scoring,return_train_score=True)
    clf.fit(X,y)
    print("Best parameters set found on development set:")
    print('%r\nTrain: %.3f Test: %.3f' % (clf.best_params_,clf.cv_results_['mean_train_score'][clf.best_index_],clf.cv_results_['mean_test_score'][clf.best_index_]))
    return clf.best_estimator_

def classifying_report(clf, X_test):
    y_pred = clf.predict(X_test)
    print
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nAccuracy: %.4f' % accuracy_score(y_test,y_pred))
    print
    print("Classification Report:")
    print(classification_report(y_test,y_pred,labels=sorted(set(y)),digits=4))

cat1 = 'cheese_plate'
cat2 = 'cheesecake'

f_train = open('{}_{}_train_set.pkl'.format(cat1,cat2),'rb')
train_data = pickle.load(f_train)

f_test = open('{}_{}_test_set.pkl'.format(cat1,cat2),'rb')
test_data = pickle.load(f_test)

desc, y = gen_sift_features(train_data)
y = y.tolist()

index_to_be_deleted = []
for i in range(len(desc)):
    if desc[i] is None: 
        index_to_be_deleted.append(i)
        
for index in index_to_be_deleted:
    del desc[index]
    del y[index]
    
n_clusters = [2,5,10,20,50,75,100,200]
inertia = []
for n_clus in n_clusters[:]:
    
    X_test = [x[0] for x in test_data]
    y_test = [x[1] for x in test_data]
    
    kmeans = KMeans(n_clusters=n_clus, n_init=20)
    
    X, cluster_model = cluster_features(desc, kmeans)
    inertia.append(cluster_model.inertia_)
    
    test_set = np.zeros((len(test_data), n_clus), np.int64)
    i = 0
    for photo in X_test:
        test_set[i] = img_to_vect(photo, cluster_model)
        i += 1
        
    test_set = test_set.tolist()
    index_to_be_deleted = []
    for i in range(len(test_set)):
        if test_set[i] == [-1, -1]: 
            index_to_be_deleted.append(i)
            
    for index in index_to_be_deleted:
        del y_test[index]
        del test_set[index]
    
#    params = [{'C': [0.001,0.01,.1,1,10],
#            'class_weight': [None,'balanced']}]
#    linearSVC_best = gridsearch(LinearSVC(),params)
#    classifying_report(linearSVC_best, test_set)

#     Gaussian SVM
#    params= [{'kernel': ['rbf'], 
#             'gamma': ['auto',1e-4,1e-3,1e-2,1e-1,1], 
#             'C': [1e-3,1e-2,.1,1,5,10],
#             'class_weight': [None,'balanced'],
#             'decision_function_shape': ['ovo','ovr']}]
#    rbfSVC_best = gridsearch(SVC(),params)
#    classifying_report(rbfSVC_best, test_set)

#    params = [{'max_depth': [2,5,10,15,20,None],
#               'splitter': ['best','random'], 
#               'presort': [True,False],
#               'criterion': ['gini'],
#               'min_samples_split': [2,5,10,15,20],
#               'max_features': ['auto',None,'log2']}]
#    dectree_best = gridsearch(DecisionTreeClassifier(),params)
#    classifying_report(dectree_best, test_set)
    
#    params = [{'n_estimators': [5,10,15,20],
#               'criterion': ['gini'],
#               'max_features': ['auto',None,'log2'],
#               'max_depth': [2,5,10,15,None],
#               'min_samples_split': [2,5,7,10,15],
#               'bootstrap': [True,False]}]
#    randforest_best = gridsearch(RandomForestClassifier(),params)
#    classifying_report(randforest_best, test_set)