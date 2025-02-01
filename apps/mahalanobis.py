import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from scipy.spatial.distance import mahalanobis

class MahalanobisDetector:

    def __init__(self, subjects, data):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        self.data = data
    
    def training(self):
        self.mean_vector = self.train.mean().values  
        
    def evaluateSet1(self):
        eers = []
        for subject in self.subjects:
            genuine_user_data = self.data.loc[self.data.subject == subject, "H" : "DD"]
            imposter_data = self.data.loc[self.data.subject != subject, :]
            lengthData = int(len(genuine_user_data)*0.7)
            self.train        = genuine_user_data[:lengthData]
            self.test_genuine = genuine_user_data[lengthData+1:]
            self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H":"DD"]
            self.training()
            self.testing()
            labels = [0]*len(self.user_scores) + [1]*len(self.imposter_scores)
            fpr, tpr, thresholds = roc_curve(labels, self.user_scores + self.imposter_scores)
        return fpr, tpr, thresholds
    
    
    def evaluateSet2(self):
        eers = []
        for subject in self.subjects:
            genuine_user_data = self.data.loc[self.data.subject == subject, "H.period":"H.Return"]
            imposter_data = self.data.loc[self.data.subject != subject, :]
            self.train        = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
            self.training()
            self.testing()
            labels = [0]*len(self.user_scores) + [1]*len(self.imposter_scores)
            fpr, tpr, thresholds = roc_curve(labels, self.user_scores + self.imposter_scores)
        return fpr, tpr, thresholds
    
    def testing(self):
        self.user_scores = []
        self.imposter_scores = []

        # Compute covariance matrix and inverse
        cov_matrix = np.cov(self.train.T)
        cov_inv = np.linalg.pinv(cov_matrix)

        for i in range(self.test_genuine.shape[0]):
            cur_score = mahalanobis(self.test_genuine.iloc[i].values, self.mean_vector,cov_inv)
            self.user_scores.append(cur_score)

        for i in range(self.test_imposter.shape[0]):
            cur_score = mahalanobis(self.test_imposter.iloc[i].values, self.mean_vector,cov_inv)
            self.imposter_scores.append(cur_score)