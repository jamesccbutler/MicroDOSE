import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from sklearn.cluster import KMeans

#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout
#from skbio.stats.composition import clr

from torch.utils.data import DataLoader, TensorDataset
import torch

import config
    

class DataCont(object):
    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None, label_dict = None, phylo=None):
        # Training and test data holders
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_dict = label_dict
        self.phylo = phylo

class Experimentor(object):
    def __init__(self, data : DataCont, select_feature = False, phylo_clust = False, test = True, select_labels = None):
        # Data name
        #self.exp_name = exp_name

        # Create directory with experiment name
        #self.result_path = os.path.join(os.getcwd(), 'Result_Files', exp_name)
        #self.model_path = os.path.join(os.getcwd(), 'Model_Files', exp_name)
        #self.data_path = os.path.join(self.result_path, 'Data_Files')
        #if not os.path.exists(self.result_path):
            #os.makedirs(self.result_path)
        #if not os.path.exists(self.model_path):
            #os.makedirs(self.model_path)
        #if not os.path.exists(self.data_path):
            #os.makedirs(self.data_path)

        # Training and test data holders
        self.test = test
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.phylo = data.phylo
        self.X_train_binary = None
        self.label_dict = data.label_dict

        if self.test == True:
            self.X_test = data.X_test
            self.y_test = data.y_test
            self.X_test_binary = None
        

        self.status = None

        # Augmented data holders
        self.aug_name = None
        self.aug_rates = None
        self.X_augs = None
        self.y_augs = None
        self.X_train_augs = None
        self.y_train_augs = None

        # Classifiers
        scoring='roc_auc'
        n_jobs=-1
        cv=5

        #self.classifiers = [SVC(probability=True, random_state=0, gamma='scale'), RandomForestClassifier(random_state=0, n_estimators=100), MLPClassifier(random_state=0, hidden_layer_sizes=(128, 64, 32), max_iter=500)]
        #self.classifier_names = ["SVM", "RF", "NN"]

        #self._remove_sparse()

        #self._standardize()

        # Feature Selection

        if select_labels is not None:
            self._label_selection(select_labels)

        if select_feature == True:
            self._select_feature()

        if phylo_clust == True:
            self._phylo_cluster()


        # Save train and test data in augmentation directory
        #if not os.path.exists(os.path.join(self.data_path, f'X_train.csv')) or not os.path.exists(os.path.join(self.data_path, f'X_test.csv')):
            #pd.DataFrame(self.X_train).to_csv(os.path.join(self.data_path, f'X_train.csv'))
            #pd.DataFrame(self.y_train).to_csv(os.path.join(self.data_path, f'y_train.csv'))
            #pd.DataFrame(self.X_test).to_csv(os.path.join(self.data_path, f'X_test.csv'))
            #pd.DataFrame(self.y_test).to_csv(os.path.join(self.data_path, f'y_test.csv'))
        
        self.make_binary()

        self.prep_diffusion()
        
        #self.x_train_inversed = self.scaler.inverse_transform(self.X_train)
        #self.x_test_inversed = self.scaler.inverse_transform(self.X_test)
        
         # Distribution
        #self.draw_histogram(Xs=[self.X_train, self.X_test], Xs_labels=["X_train", "X_test"])

        # Viz
        #self.visualize_featurewise_wss()
        #self.visualize_samplewise_wss()
        #self.visualize(X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test)

    def _label_selection(self, select_labels):
        # filter for disease

        y_tr = np.asarray(self.y_train).flatten()
        y_te = np.asarray(self.y_test).flatten()

        train_mask = (y_tr == select_labels[0]) | (y_tr == select_labels[1])
        val_mask = (y_te == select_labels[0]) | (y_te == select_labels[1])

        self.X_train = self.X_train[train_mask]
        self.y_train = self.y_train[train_mask]
        self.X_test = self.X_test[val_mask]
        self.y_test = self.y_test[val_mask]

        # Create new label mapping: Assign 0 and 1 to selected labels
        new_label_dict = {select_labels[0]: 0, select_labels[1]: 1}

        # Apply new labels using mapping
        self.y_train = np.vectorize(new_label_dict.get)(self.y_train)
        self.y_test = np.vectorize(new_label_dict.get)(self.y_test)

        # Update self.label_dict to reflect the new numeric-to-string mapping
        self.label_dict = {0: self.label_dict[select_labels[0]], 1: self.label_dict[select_labels[1]]}

    def prep_diffusion(self):
        self.X_train = torch.tensor(self.X_train).unsqueeze(1).to(torch.float32)
        self.y_train = torch.tensor(self.y_train).to(torch.float32)
        self.phylo = self.phylo
        self.X_train_binary = torch.tensor(self.X_train_binary).unsqueeze(1).to(torch.float32)
        if self.test == True:
            self.X_test = torch.tensor(self.X_test).unsqueeze(1).to(torch.float32)
            self.y_test = torch.tensor(self.y_test).to(torch.float32)
            self.X_test_binary = torch.tensor(self.X_test_binary).unsqueeze(1).to(torch.float32)


        # End of __init__
    def make_feature_rows(self, df):
        X_split_df = pd.DataFrame(df)
        X_split_df.columns = self.phylo.columns
        X_split_df = pd.concat([self.phylo, X_split_df]).T
        X_split_df.columns = X_split_df.columns.astype(str)
        X_split_df.columns.values[0] = "clad_name"
        X_split_df["clad_name"] = X_split_df.index.astype(str)  + "|" + X_split_df["clad_name"]
        X_split_df = X_split_df.reset_index(drop = True)
        return(X_split_df)

    def make_binary(self):
        self.X_train_binary = np.where(self.X_train != 0, 1, 0)
        if self.test == True:
            self.X_test_binary = np.where(self.X_test != 0, 1, 0)
    
    def phylo_clust(self, data):
        data = self.make_feature_rows(data)
        otu_column_name = data.columns.tolist()[0]
        phylum_list = []

        for i in range(len(data)) :
            phylum_list.append(data[otu_column_name][i].split('|')[1])

        data["phylum"] = phylum_list 

        phylum_uniq_list = data["phylum"].unique().tolist()
        phylum_otu_count_list = []

        for i in range(len(phylum_uniq_list)) :
            tmp_phylum = data[data["phylum"] == phylum_uniq_list[i]]
            phylum_otu_count_list.append(len(tmp_phylum))

        phylum_otu_count_df = pd.DataFrame({'phylum' : phylum_uniq_list, 'otu_count' : phylum_otu_count_list})
        phylum_otu_count_df = phylum_otu_count_df.sort_values(by="otu_count", ascending = False)
        phylum_otu_count_df.reset_index(inplace = True, drop = True)

        final_data_df = pd.DataFrame()

        for i in range(len(phylum_otu_count_df)) : 
            tmp_phylum = data[data["phylum"] == phylum_otu_count_df["phylum"][i]]
            tmp_phylum.set_index(otu_column_name, inplace = True, drop= True)
            del tmp_phylum["phylum"]
            colnames = tmp_phylum.columns.tolist()
            tmp_phylum = tmp_phylum.T
            tmp_phylum_corr = tmp_phylum.astype('float64').corr(method = 'spearman')
            tmp_phylum_otu_list = tmp_phylum_corr.index.tolist()
            tmp_phylum_otu_corr_list = []
            for otu in range(len(tmp_phylum_otu_list)) :
                tmp_phylum_otu_row_abs = tmp_phylum_corr.iloc[otu].abs()
                tmp_cumulative_corr = 1
                for j in range(len(tmp_phylum_otu_row_abs)) :
                    if tmp_phylum_otu_row_abs[j] != 0.0 :
                        tmp_cumulative_corr *= tmp_phylum_otu_row_abs[j]
                tmp_phylum_otu_corr_list.append(tmp_cumulative_corr ** (1/len(tmp_phylum_otu_row_abs)))
            tmp_phylum_otu_corr_df = pd.DataFrame({'otu' : tmp_phylum_otu_list, 'corr' : tmp_phylum_otu_corr_list})
            tmp_phylum_otu_corr_df = tmp_phylum_otu_corr_df.sort_values(by = 'corr', ascending = False)
            tmp_data_based_on_corr = pd.merge(tmp_phylum_otu_corr_df, data, left_on = "otu", right_on = otu_column_name)
            del tmp_data_based_on_corr["corr"]
            #del tmp_data_based_on_corr[otu_column_name]
            del tmp_data_based_on_corr["phylum"]
            tmp_data_based_on_corr["cluster"] = i
            final_data_df = pd.concat([final_data_df, tmp_data_based_on_corr], axis = 0)
        return(final_data_df)
    
    def _standardize(self):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(self.X_train)
        if self.test == True:
            self.X_train = self.scaler.transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
   
    def _remove_sparse(self):
        X_train_cols = pd.DataFrame(self.X_train)
        
        if self.test == True:
            X_test_cols = pd.DataFrame(self.X_test)
            X_test_cols.columns = self.phylo.columns

        X_train_cols.columns = self.phylo.columns
        microbiome_df = X_train_cols
        num_samples = len(microbiome_df)
        to_drop = []
        for microbe in microbiome_df.columns.values:
            present_in = sum(microbiome_df[microbe] > 0.0000)
            if present_in <= 0.05 * num_samples:
                to_drop.append(microbe)
        microbiome_df = microbiome_df.drop(to_drop, axis=1)
        to_drop = []
        for microbe in microbiome_df.columns.values:
            mean_presence = np.mean(microbiome_df[microbe])
            if mean_presence <= 0.0001:
                to_drop.append(microbe)
        microbiome_df = microbiome_df.drop(to_drop, axis=1)
        phylo_filt = microbiome_df.columns
        self.X_train = X_train_cols.loc[:,phylo_filt].to_numpy()
        if self.test == True:
            self.X_test = X_test_cols.loc[:,phylo_filt].to_numpy()
        self.phylo = self.phylo.loc[:,phylo_filt]
        
    #def _clr_transform(self):
        #data_clr = self.make_feature_rows(self.X_train)
        #sample_list = data_clr.columns.tolist()
        #sample_list.pop(0)
        #tmp_min = 100
        #for sample in sample_list :
            #for i in range(len(data_clr)) :
                #if (tmp_min > data_clr[sample][i]) and (data_clr[sample][i] != 0.0) :
                    #tmp_min = data_clr[sample][i] 
            #pseudo_count = tmp_min/2
        #for sample in sample_list :
            #for i in range(len(data_clr)) :
                #if (data_clr[sample][i] == 0.0) :
                    #data_clr[sample][i] += pseudo_count

        #data_clr.set_index(data_clr.columns[0], inplace = True)
        #ata_clr = data_clr.T 
        #data_mat = data_clr.to_numpy(dtype="float64")
        #clr_data = pd.DataFrame(clr(data_mat))
        #clr_data.columns = data_clr.columns 
        #clr_data.index = data_clr.index
        #self.X_train = clr_data.to_numpy()
        #if self.test == True:
            #data_clr = self.make_feature_rows(self.X_test)
            #sample_list = data_clr.columns.tolist()
            #sample_list.pop(0)
            #for sample in sample_list :
                #for i in range(len(data_clr)) :
                    #if (data_clr[sample][i] == 0.0) :
                        #data_clr[sample][i] += pseudo_count

            #data_clr.set_index(data_clr.columns[0], inplace = True)
            #data_clr = data_clr.T 
            #data_mat = data_clr.to_numpy(dtype="float64")
            #clr_data = pd.DataFrame(clr(data_mat))
            #clr_data.columns = data_clr.columns
            #clr_data.index = data_clr.index
            #self.X_test = clr_data.to_numpy()
            
    def _phylo_cluster(self):
        train_phylo = self.phylo_clust(self.X_train)
        if self.test == True:
            test_phylo = self.make_feature_rows(self.X_test)
            test_phylo = pd.merge(train_phylo.loc[:,["clad_name"]], test_phylo, on='clad_name', how = "left")
        #print(train_phylo["clad_name"].dtype)
        #train_phylo["clad_name"] = train_phylo["clad_name"].astype(str)
        #split_strings = train_phylo["clad_name"].str.split('|', 1)
        #print(split_strings)
        
        #second_part = split_strings.str[1]
        #print(second_part)
        
        #phylo_new = pd.DataFrame(second_part).T.reset_index(drop = True)

        phylo_new = pd.DataFrame(train_phylo["clad_name"].str.split('|', n = 1).str[1]).T.reset_index(drop = True)
        phylo_new.columns = train_phylo["clad_name"].str.split('|').str[0].values
        self.phylo = phylo_new
        self.X_train = train_phylo.drop(columns = ["otu", "clad_name", "cluster"]).apply(pd.to_numeric).to_numpy().T
        if self.test == True:
            self.X_test = test_phylo.drop(columns = ["clad_name"]).apply(pd.to_numeric).to_numpy().T

    def _select_feature(self, n_estimators=500, max_features=256):
        clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion="entropy", random_state=0)
        clf = clf.fit(self.X_train, self.y_train)
        #print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True, max_features=max_features)
        self.X_train = model.transform(self.X_train)
        if self.test == True:
            self.X_test = model.transform(self.X_test)
        status = model.get_support()
        self.selected_feature_indices = np.where(status)[0]
        self.phylo = self.phylo[self.phylo.columns[status]]

    def draw_histogram(self, Xs : list, Xs_labels : list):
        num_Xs = len(Xs)
        fig, axs = plt.subplots(num_Xs, figsize=(15, 4 * num_Xs))
        for i in range(num_Xs):
            axs[i].hist(Xs[i].flatten(), bins='auto')
            axs[i].text(x=0.02, y=0.9, s=Xs_labels[i], transform=axs[i].transAxes)
        if self.aug_name == None: 
            self.aug_name = 'beforeAug'
        plt.savefig(os.path.join(self.result_path, self.aug_name + 'Dist.png'))

    def visualize(self, X_train, y_train, X_test, y_test, lower_bound=-5, upper_bound=5):
        # Sort data by class label
        idx = np.argsort(y_train)
        X_train_sorted = X_train[idx]
        y_train_sorted = y_train[idx]
        idx = np.argsort(y_test)
        X_test_sorted = X_test[idx]
        y_test_sorted = y_test[idx]

        # Get the number of unique class labels and the number of plots
        classes = np.unique(y_train)
        num_class = len(classes)
        num_plots = num_class * 2

        fig, axs = plt.subplots(num_plots, figsize=(15,4*num_plots))
        
        for i in range(0, num_plots):
            if i < (num_plots / 2):
                # Training data viz
                ms = axs[i].matshow(X_train_sorted[y_train_sorted == classes[i]], cmap="seismic", aspect='auto')
                ms.set_clim(lower_bound, upper_bound)
            else:
                # Test data viz
                ms = axs[i].matshow(X_test_sorted[y_test_sorted == classes[i - num_class]], cmap="seismic", aspect='auto')
                ms.set_clim(lower_bound, upper_bound)

        # Location and size of colorbar: [coordinate1, coordinate2 inthe figure, colorbar width, height]
        cax = fig.add_axes([0.94, 0.2, 0.02, 0.4])
        
        # Add colorbar
        fig.colorbar(mappable=ms, cax=cax, extend='both')

        # Show figure
        plt.show()

    def visualize_aug(self, X_train, y_train, X_test, y_test, X_aug, y_aug, lower_bound=-5, upper_bound=5):
        # Sort data by class label
        idx = np.argsort(y_train)
        X_train_sorted = X_train[idx]
        y_train_sorted = y_train[idx]
        idx = np.argsort(y_test)
        X_test_sorted = X_test[idx]
        y_test_sorted = y_test[idx]
        idx = np.argsort(y_aug)
        X_aug_sorted = X_aug[idx]
        y_aug_sorted = y_aug[idx]

        # Get the number of unique class labels and the number of plots
        classes = np.unique(y_train)
        num_class = len(classes)
        num_plots = num_class * 3

        fig, axs = plt.subplots(num_plots, figsize=(15,4*num_plots))

        viz_chunk = num_plots / 3
        
        for i in range(0, num_plots):
            if i < (viz_chunk):
                # Training data viz
                ms = axs[i].matshow(X_train_sorted[y_train_sorted == classes[i]], cmap="seismic", aspect='auto')
                ms.set_clim(lower_bound, upper_bound)
            elif i < (viz_chunk*2):
                # Test data viz
                ms = axs[i].matshow(X_test_sorted[y_test_sorted == classes[i - num_class]], cmap="seismic", aspect='auto')
                ms.set_clim(lower_bound, upper_bound)
            else:
                # Aug data viz
                ms = axs[i].matshow(X_aug_sorted[y_aug_sorted == classes[i - num_class*2]], cmap="seismic", aspect='auto')
                ms.set_clim(lower_bound, upper_bound)

        # Location and size of colorbar: [coordinate1, coordinate2 inthe figure, colorbar width, height]
        cax = fig.add_axes([0.94, 0.2, 0.02, 0.4])
        
        # Add colorbar
        fig.colorbar(mappable=ms, cax=cax, extend='both')

        # Show figure
        plt.savefig(os.path.join(self.result_path, self.aug_name + 'Viz.png'))

    def visualize_featurewise_wss(self):
        # Helper func
        def calculate_WSS(points, kmax):
            sse = []
            for k in range(1, kmax+1):
                kmeans = KMeans(n_clusters = k, random_state=1).fit(points)
                centroids = kmeans.cluster_centers_
                pred_clusters = kmeans.predict(points)
                curr_sse = 0
                # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
                for i in range(len(points)):
                    curr_center = centroids[pred_clusters[i]]
                    curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
                sse.append(curr_sse)
            return sse
        fig = plt.figure()
        plt.plot([k for k in range(1, 10+1)], calculate_WSS(self.X_train.T, 10))
        plt.savefig(os.path.join(self.result_path, 'featurewise_WSS.png'))

    def visualize_samplewise_wss(self):
        # Helper func
        def calculate_WSS(points, kmax):
            sse = []
            for k in range(1, kmax+1):
                kmeans = KMeans(n_clusters = k, random_state=1).fit(points)
                centroids = kmeans.cluster_centers_
                pred_clusters = kmeans.predict(points)
                curr_sse = 0
                # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
                for i in range(len(points)):
                    curr_center = centroids[pred_clusters[i]]
                    curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
                sse.append(curr_sse)
            return sse
        fig = plt.figure()
        plt.plot([k for k in range(1, 10+1)], calculate_WSS(self.X_train, 10))
        plt.savefig(os.path.join(self.result_path, 'samplewise_WSS.png'))
        
    def classify_without_augmentation(self):
        with open(os.path.join(self.result_path, 'noAug.txt'), "w") as f:

            # Write result header
            f.write("Clf\tAUROC\tAUPRC\tACC  \tREC  \tPRE  \tF1  \n")

            for clf, clf_name in zip(self.classifiers, self.classifier_names):
                clf.fit(self.X_train, self.y_train)
                print(f'Best parameter on training set: {clf.best_params_}')
                y_pred = clf.predict(self.X_test)
                y_prob = clf.predict_proba(self.X_test)

                precisions, recalls, _ = precision_recall_curve(self.y_test, y_prob[:, 1])

                # Performance Metrics : AUROC, AUPRC, ACC, Recall, Precision, F1
                auroc = round(roc_auc_score(self.y_test, y_prob[:, 1]), 3)
                auprc = round(auc(recalls, precisions), 3)
                acc = round(accuracy_score(self.y_test, y_pred), 3)
                rec = round(recall_score(self.y_test, y_pred), 3)
                pre = round(precision_score(self.y_test, y_pred), 3)
                f1 = round(f1_score(self.y_test, y_pred), 3)
                
                f.write(f"{clf_name}\t{auroc}\t{auprc}\t{acc}\t{rec}\t{pre}\t{f1}\n")

    def classify_with_non_DL_augmentation(self):
         # Time stamp
        start_time = time.time()

        with open(os.path.join(self.result_path, self.aug_name + 'Aug.txt'), "w") as f:
            # Write result header
            f.write("Clf\tAugRate\tAUROC\tAUPRC\tACC  \tREC  \tPRE  \tF1  \n")

            for clf, clf_name in zip(self.classifiers, self.classifier_names):
                # Get best params from training data
                clf.fit(self.X_train, self.y_train)
                best_est = copy.deepcopy(clf.best_estimator_)

                for i in range(len(self.aug_rates)):
                    print(f'aug_rate: {self.aug_rates[i]}')
                    # Fit on augmented training data
                    best_est.fit(self.X_train_augs[i], self.y_train_augs[i])
                    y_pred = best_est.predict(self.X_test)
                    y_prob = best_est.predict_proba(self.X_test)

                    precisions, recalls, _ = precision_recall_curve(self.y_test, y_prob[:, 1])

                    # Performance Metrics : AUROC, AUPRC, ACC, Recall, Precision, F1
                    auroc = round(roc_auc_score(self.y_test, y_prob[:, 1]), 3)
                    auprc = round(auc(recalls, precisions), 3)
                    acc = round(accuracy_score(self.y_test, y_pred), 3)
                    rec = round(recall_score(self.y_test, y_pred), 3)
                    pre = round(precision_score(self.y_test, y_pred), 3)
                    f1 = round(f1_score(self.y_test, y_pred), 3)

                    f.write(f"{clf_name}\t{self.aug_rates[i]}\t{auroc}\t{auprc}\t{acc}\t{rec}\t{pre}\t{f1}\n")
        
        print(f"--- Classified with {self.aug_name} augmentation in {round(time.time() - start_time, 2)} seconds ---")

    def classify_with_wGAN_augmentation(self, fixed_num_gans = None):
         # Time stamp
        start_time = time.time()

        if fixed_num_gans:
            g_range = range(fixed_num_gans-1, fixed_num_gans)
        else:
            g_range = range(len(self.X_augs))

        with open(os.path.join(self.result_path, self.aug_name + f'Aug.txt'), "w") as f:
            # Write result header
            f.write("NumGANs\tClf  \tAugRate\tAUROC\tAUPRC\tACC  \tREC  \tPRE  \tF1  \n")
            for clf, clf_name in zip(self.classifiers, self.classifier_names):
                # Get best params from training data
                clf.fit(self.X_train, self.y_train)
                best_est = copy.deepcopy(clf.best_estimator_)

                for g in g_range:
                    for i in range(len(self.aug_rates)):
                        print(f'aug_rate: {self.aug_rates[i]}, # of GANs: {g+1}')
                        # Fit on augmented training data
                        best_est.fit(self.X_train_augs[g][i], self.y_train_augs[g][i])
                        y_pred = best_est.predict(self.X_test)
                        y_prob = best_est.predict_proba(self.X_test)

                        precisions, recalls, _ = precision_recall_curve(self.y_test, y_prob[:, 1])

                        # Performance Metrics : AUROC, AUPRC, ACC, Recall, Precision, F1
                        auroc = round(roc_auc_score(self.y_test, y_prob[:, 1]), 3)
                        auprc = round(auc(recalls, precisions), 3)
                        acc = round(accuracy_score(self.y_test, y_pred), 3)
                        rec = round(recall_score(self.y_test, y_pred), 3)
                        pre = round(precision_score(self.y_test, y_pred), 3)
                        f1 = round(f1_score(self.y_test, y_pred), 3)

                        f.write(f"{g+1}\t{clf_name}\t{self.aug_rates[i]}\t{auroc}\t{auprc}\t{acc}\t{rec}\t{pre}\t{f1}\n")
        
        print(f"--- Classified with {self.aug_name} augmentation in {round(time.time() - start_time, 2)} seconds ---")

    def _pred_with_optimal_threshold(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        y_pred_optima = y_prob[:, 1] > thresholds[ix]
        return y_pred_optima