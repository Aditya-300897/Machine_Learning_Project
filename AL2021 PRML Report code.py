#!/usr/bin/env python
# coding: utf-8

# ## Emotion recognition through EEG analysis using ML algorithms
# 
# The following notebook includes the code for the Assignment 3 of 11512 PRML unit. Authors: Aditya Salvi - u3230853, Lilit Griggs-Atoyan - u3182730.
# 
# The purpose of this work is to find out whether Kbest or MI, as filter methods, are helping to improve the classification models' performances.
# 
# Content for this notebook is as follows:

# Table of Contents
#  
# * [Section 1. Exploratory Data Analysis](#section1)
#     * [Section 1.1. Data Inspection](#section_1_1)
#     * [Section 1.2. Data Visualization](#section_1_2)
#     * [Section 1.3. Conclusion](#section_1_3)
# * [Section 2. Pre-Processing](#section2)
#     * [Section 2.1](#section_2_1)
#     * [Section 2.2](#section_2_2)
# * [Section 3. Applying Models](#section3)
#     * [Section 3.1. SVM model](#section_3_1)
#         * [Section 3.1.1.Original dataset](#section_3_1_1)
#         * [Section 3.1.2 F value Kbest Dataset](#section_3_2_2)
#         * [Section 3.1.3 MI dataset](#section_1_2_3)
#     * [Section 3.2. Random Forest model](#section_3_2)
#         * [Section 3.2.1.Original dataset](#section_3_1_1)
#         * [Section 3.2.2 Kbest Dataset](#section_1_2_2)
#         * [Section 3.2.3 MI dataset](#section_1_2_3)
#     * [Section 3.3. Artificial Neural Network (MLP](#section_3_3)
# 

# ## Section 1. Exploratory Data Analysis <a class="anchor" id="section1"></a>

# Firstly, let's import data. 
# For that we need to load some libraries for setting up a working directory (os), data manipulation (pandas), and supporting operations with arrays (numpy).

# In[56]:


import os 
import sys
import pandas as pd
import numpy as np


# In[58]:


from os import chdir, getcwd
 
    


# In[265]:


# This is for loading the data from the path.

# path = sys.argv[1]

# df = pd.read_csv(path)

# https://www.kaggle.com/birdy654/eeg-brainwave-dataset-feeling-emotions/download


# In[59]:


#Change directory to the desired one
os.chdir('...11512 Pattern Rec and Machine Learn/project proposal')


# In[60]:


df= pd.read_csv('emotions.csv/emotions.csv')


# ### Section 1.1. Data Inspection <a class="anchor" id="section_1_1"></a>

# In[11]:


df.head()  #get the top 5 rows (by default is 5), dataset has 2459 columns with 'label' being the dependent variable


# In[12]:


print( "The number of rows is:", df.shape[0])  # get the overall look of the data
print( "The number of columns is:", df.shape[1])


# In[13]:


df.describe()  # get the summary statistics. We can notice that data is of various ranges. 
#Raises a need to do data transformation.Also, we notice that Label column cannot be worked with and we need to encode it.


# In[62]:


df['label'].unique()  # check how many classes are there by getting the unique values for Label


# In[63]:


df.isnull().sum().sum() #check for the null values.No null values in the dataset. (We already knew this from the way the feature extraction was performed.)


# In[64]:


df.info()  #one last glipms on the dataset, 
#the data are of type float64 which means we won't need to change the type of data.


# ### Section 1.2. Data Visualization <a class="anchor" id="section_1_2"></a>

# In[65]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random


# In[66]:


df_sub1= df.sample(n=4,axis='columns')  #randomly choose 4 columns to visualize. 
#We are doing simple histograms for quick vizualisation.
#Every time we run the code we get distributions of different columns
df_sub1.hist()         # We can see from the histograms as well, that our data needs to be normalized.


# In[67]:


df.label.hist()  # a quick check to see if the data is balanced, 
#as we can see the labels are distributed almost equally amongs the classes.
# there is no need to balance our dataset.


# In[68]:


# Just for the following visualization task, we do label encoding here. We will perform label encoding again, as part of data preprocessing.
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()  #instantiating label encoder object and applying it to label column
label = enc.fit_transform(df.label)


# In[69]:


label   #we can see 0 is for Negative emotions, 1 is for Neutral, and 2 is for Positive emotions


# In[70]:


label=pd.DataFrame(label, index= df.label.index, columns=['label']) #creating one-column dataframe for label to merge


# In[97]:


# create another dataset with two random columns to visualize scatterplot.
df_sub2 = pd.merge(df.iloc[:, random.sample(range(0, 2547), 2)], label, left_index=True, right_index=True)

#define the scatterplot object and draw the scatter for two variables
fig, ax = plt.subplots()

scatter = ax.scatter(df_sub2.iloc[:, 0], df_sub2.iloc[:, 1], c = df_sub2.label) #give the x,y and c parameters

# create a legend with unique colors from the scatter 
#(Source for this line: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html)
legend = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Classes")
ax.add_artist(legend)

ax.set_xlabel(df_sub2.iloc[:, 0].name)
ax.set_ylabel(df_sub2.iloc[:, 1].name) 
ax.set_title("Scatterplot")
plt.show()

# Run this code as many times as you like to see scatterplot of two randomly chosen variables


# In[98]:


# Visualize the correlation matrix. There are some highly correlated features.
import matplotlib.pyplot as plt

# plt.matshow(X_scaled_df.corr())
# plt.show()

f = plt.figure(figsize=(10, 10))
plt.matshow(df.drop(columns=['label']).corr(), fignum=f.number) # Drop label column and visualize the correlation matrix
# plt.xticks(range(df.drop(columns=['label']).shape[1]), df.drop(columns=['label']).columns, fontsize=3, rotation=45)
# plt.yticks(range(df.drop(columns=['label']).shape[1]), df.drop(columns=['label']).columns, fontsize=3)
cb = plt.colorbar(shrink=0.75)
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=16)


# ### Section 1.3. Conclusions from Exploratory Data Analysis <a class="anchor" id="section_1_3"></a>
# 
# 1. Dataset is balanced.
# 2. Dataset does not have null values.
# 3. There are highly skewed variables in the dataset.
# 3. There are some variables in the dataset with values that are considerably far apart from each other.
# 4. We need to perform normalization of the data to fix the problem in 3rd point.
# 5. Scatter plot shows that there are some highly correlated and not so correlated variables in the dataset, meaning that we can use feature selection techniques to remove reduntandant variables.

# # Section 2. Pre-Processing <a class="anchor" id="section2"></a>

# From the previous section we found out that we need to standardize our data. 
# Also, because our dataset is massive, we are going to apply feature selection techniques--Kbest and Mutual Information(MI)-- to create a subset of the dataset and work with those subsets. We are trying to follow the original work by Bird et al, hence we will choose datasets with 63 variables, like they have done. We will then compare the the evaluation metrics of the models applied to the original dataset vs the subsets obtained from this step of feature selection.
# 
# *Note: We are going to repeat label encoding to avoid any issues arising from already used code.

# First, we split our dataset to variables and independent variable.

# In[100]:


X = df.drop(columns=['label']) #dropping the label column
y = df['label']
print("X:", X.shape)
print("y:", y.shape)
X.head(3)


# ## Section 2.1. Standardizing data <a class="anchor" id="section_2_1"></a>

# In[101]:


from sklearn.preprocessing import StandardScaler # for standardizing our data
from sklearn.preprocessing import LabelEncoder # for encoding our label column


# In[102]:


scaler = StandardScaler() #instantiating standard scaler object
X_scaled = scaler.fit_transform(X.values) #applying standard scaler on X
X_scaled_df=pd.DataFrame (X_scaled, index= X.index, columns=X.columns) # for future use
print(X_scaled.shape)
print("Features mean: %.2f, Features variance: %.2f" % ((np.mean(X_scaled)),(np.std(X_scaled))))
# checking whether the transformed data has mean=0 and std=1)


# In[103]:


enc = LabelEncoder()  #instantiating label encoder eobject and applying it to y
y_enc = enc.fit_transform(y) # fitting the encoder on y
y_enc_df=pd.DataFrame(y_enc, index= y.index, columns=['label']) # turning y into df to merge with features for future use
y_enc_df.shape


# In[104]:


sdf=pd.merge(X_scaled_df, y_enc_df, left_index=True, right_index=True) #creating the standardized and scaled dataset


# ## Section 2.2. Obtaining training and testing sets <a class="anchor" id="section_2_2"></a>

# In[105]:


# Splitting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(sdf.drop(['label'], axis=1),y_enc,
                                                 test_size=0.3,
                                                 random_state=0)


# In[106]:


# Creating dataframes from the independent variables and output variable to merge in a new dataframe
X_train_df=pd.DataFrame(X_train, index= X_train.index, columns=X_train.columns.values)
y_train_df=pd.DataFrame(y_train, index=X_train.index, columns=['label'])
# X_train_df.head()
y_train_df.head()


# ## Section 2.3. Feature Selection<a class="anchor" id="section_2_3"></a>

# By the end of this Section we want have two differently assessed datasets X_train_MI, and X_train_F, both containg 63 variables to follow the original work that was done on this dataset. 
# 
# We have chosen **filter methods** for feature selection, because they are algorithm-independent which makes them computationally simple and fast, as well as they are able to handle high dimensional datasets. 

# In[107]:


# Importing feature selection objects for classification tasks
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# ### Section 2.3.1. Feature Selection via F score <a class="anchor" id="section_2_3_1"></a>

# Filter
# Wrapper
# Embedded methods
# 

# In[108]:


# Instantiating the selector based on ANOVA F-value scores and apply to the train set.
select_f = SelectKBest(f_classif, k=63).fit(X_train.values, y_train) 
columns=X_train.columns[select_f.get_support()] # get the names of the columns from the previous
df_f= select_f.transform(X_train)  # apply select_f object to pick the best variables in X_train
X_train_F= pd.DataFrame(df_f, columns=columns)  # create a subset of X_train with the chosen 63 columns
X_test_F= pd.DataFrame(X_test, columns=columns) # create a subset of X_test with the chosen 63 columns


# ### Section 2.3.2 Feature Selection via Mutual Information (MI) <a class="anchor" id="section_2_3_2"></a>

# Information gain or mutual information  is a filter feature selection technique. It assesses the ability of the independent variables to predict the output variable.
# 
# 

# In[109]:


# Apply mutual info algorithm on the train set 
mutual_info= mutual_info_classif(X_train, y_train, random_state=42) 


# In[110]:


mutual_info = pd.Series(mutual_info)  # convert array of scores into series 
mutual_info.index = X_train.columns   # assign the same incides as for X_train
mutual_info.sort_values(ascending=False)  # sort values in descending order
mutual_info
select_mi = SelectKBest(mutual_info_classif, k=63).fit(X_train.values, y_train)  # instantiate selection
# method to select best 63 features. Apply the selection object on the training set


# In[111]:


columns=X_train.columns[select_mi.get_support()] # get the names of the columns from the previous
df_mi= select_mi.transform(X_train.values)
X_train_MI= pd.DataFrame(df_mi, columns=columns)
X_test_MI= pd.DataFrame(X_test, columns=columns)


# # Section 3. Applying ML models <a class="anchor" id="section3"></a>

# Since we  have three classes, namely Negative, Neutral, Positive, our problem is a multiclass classification. Hence, when doing the classification task, we have chosen the models and parameters designed for multiclass classification.

# ## Section 3.1. SVM <a class="anchor" id="section_3_1"></a>
# 

# The multiclass support is handled according to a one-vs-one scheme.

# In[112]:


from sklearn.svm import SVC 
from sklearn.svm import LinearSVC  #to handle modelling on large datasets
from sklearn.model_selection import GridSearchCV  # for parameter tuning
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns  # for visualization


# For accuracy calculations
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix


# ### 3.1.1. LinearSVC: SVM on the whole dataset   <a class="anchor" id="section_3_1_1"></a>
# 

# ##### a) One-vs-Rest wrapper with LinearSVC classifier 

# In[134]:


# Apply One-vs-Rest wrapper on LinearSVC classifier (using LinearSVC because of the large dataset)
clf = OneVsRestClassifier(LinearSVC(max_iter=10000,dual=False, random_state=0, tol=0.00001,C=0.1, class_weight= "balanced")).fit(X_train, y_train) # the model wasn't converging, and needed more iteration to be done.

# Predict the y values based on the unseen data
y_pred_lsvc= clf.predict(X_test)

# Operate classification report function
clr = classification_report(y_test, y_pred_lsvc) 

# Print the results
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_lsvc))
print("Precision:",metrics.precision_score(y_test, y_pred_lsvc, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_lsvc, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_lsvc, average = 'weighted'))
print("Classification Report SVM on original data:\n----------------------\n", clr)

# Note: this output takes a while to load. It wasn't converging with 5000 iterations, so we had to increase the number to 10000.


# ##### b) One-vs-One LinearSVC classifier 

# The code below is just another way of trying LinearSVC, which uses one-vs-one approach for classification. We have used different parameters in the example below, but if we use those parameters in the One-vc-Rest version, the results were almost identical. Although we didn't check the time, but b) version seemed to take longer to train and predict than version a).

# In[125]:


#Use LInearSVC() classifier with default one-vs-one algorithm
# Note* The parameters chosen here are different to the first model

# clf2 = LinearSVC(max_iter=10000,random_state=0, tol=0.00001, class_weight= "balanced").fit(X_train, y_train) # the model wasn't converging, and needed more iteration to be done.
# y_pred2= clf2.predict(X_test)

# clf2 = LinearSVC()
# clf2.fit(X_train_MI, y_train)
# clr2 = classification_report(y_test, y_pred2)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Classification Report SVM on original data:\n----------------------\n", clr2)


# In[201]:


#clf1.get_params()


# **c) Hyperparameter tuning for LinearSVC**

# In[136]:


# Passing on the values for GridSearch parameters  # this output takes at least three minutes to load
parameters_lsvc = {'C': [1.0, 0.1, 0.001, 0.5], 'max_iter': [8000, 10000]}
grid_search_lsvc = GridSearchCV(LinearSVC(), parameters_lsvc, cv=10, n_jobs=-1)
grid_search_lsvc.fit(X_train, y_train)   # fitting gridsearch on the train set


# In[148]:


# grid_search_lsvc.get_params()


# In[137]:


print(" Results from Grid Search " )

print("\n The best estimator across ALL searched params:\n",grid_search_lsvc.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_search_lsvc.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_search_lsvc.best_params_)
# Code source: https://www.projectpro.io/recipes/find-optimal-parameters-using-gridsearchcv#:~:text=%20How%20to%20find%20optimal%20parameters%20using%20GridSearchCV,have%20a%20look%20on%20the%20important...%20More%20


# #### d) Using the best model obtained from GridSearch to make prediction

# In[138]:


y_pred_lsvc_best=grid_search_lsvc.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_lsvc_best))
print("Precision:",metrics.precision_score(y_test, y_pred_lsvc_best, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_lsvc_best, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_lsvc_best, average = 'weighted'))


# In[149]:


print("Classification Report SVM on original data:\n----------------------\n", classification_report(y_test, y_pred_lsvc_best))


# ##### Conclusion 
# The results of the 10-fold corss-validation grid search showed that by trial and error what we had chosen for the initial model was good enough parameters to start with.
# The only problem with LinearSVC with max_iter=5000 was that the model wasn't converging, but it is now fixed having the max_iter=8000 from the best model.

# ### 3.1.2. SVC: SVM on whole dataset.  <a class="anchor" id="section_3_1_2"></a>

# We will now apply SVC classifier to compare the results with the Linear SVC.

# In[140]:


svc = SVC(random_state=0).fit(X_train, y_train)  # Chose the simple option to compare

# Predict the y values based on the unseen data
y_pred_svc= svc.predict(X_test)

#Print evaluation metrics

print( "Evaluation metrics for original SVC model")

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc))
print("Precision:",metrics.precision_score(y_test, y_pred_svc, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_svc, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_svc, average = 'weighted'))

# Print classification report

print("Classification Report SVM on original data:\n----------------------\n", classification_report(y_test, y_pred_svc))


# #### Tuning hyperparameters for SVC on whole dataset

# In[174]:


from sklearn.model_selection import GridSearchCV
# Hyperparameter tuning
parameters_svc = {'kernel':['linear', 'rbf'], 
                  'C':[1, 0.5, 0.1, 10], 
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05, 0.5],
                  'class_weight': ['balanced']}

# Instantiating gridsearch object on the classifier
grid_search_svc = GridSearchCV(SVC(random_state=0), parameters_svc, cv=10, n_jobs=-1)


# In[175]:


# Applying the best hyperparameter model on the train data  #This output takes way too long to come
grid_search_svc.fit(X_train, y_train)


# In[176]:


# Creating a model with best parameters
model_svc_best=grid_search_svc.best_estimator_

# Applying model to test set to predict y
y_pred_svc_best=model_svc_best.predict(X_test)

# Print evaluation metrics
print(" Evaluation metrics for the best SVC model")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc_best))
print("Precision:",metrics.precision_score(y_test, y_pred_svc_best, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_svc_best, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_svc_best, average = 'weighted'))


# Print the confusion matrix of the best performing SVC model
mat = confusion_matrix(y_test, y_pred_svc_best)
plt.figure(figsize=(8,6))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')

# Code source: Tutorial materials from week 7


# In[177]:


print("\n The best parameters across ALL searched params:\n",grid_search_svc.best_params_)


# In[178]:


print("Classification Report SVM on original data:\n----------------------\n", classification_report(y_test, y_pred_svc_best))


# ### Conclusion from fitting and tuning SVC on data
# 
# As we can see from 3.1.1 and 3.1.2 results, the basic SVC model with the following hyperparameters {'C': 10, 'class_weight': 'balanced', 'gamma': 0.0001, 'kernel': 'rbf'} performed best. Now we are going to apply those results on the X_train_F and X_train_MI sets and evaluate the performance of this model on those subsets.
# 
# *Note: We have already tested non-tuned models on F and MI subsets and got substantially lower accuracy and F1 score. Thus, it will be useful to just test if the performance of the best SVC model on the subsets of features will get a bit closer to that of that on the original dataset.

# ### 3.1.3. SVC on subset obtained through ANOVA F-value 

# In[151]:


#Train the best hyperparameter model on the best F score training sets.
grid_search_svc.fit(X_train_F, y_train)

# Creating a model with best parameters
model_best_F=grid_search_svc.best_estimator_

# Applying model to get predicted y
y_pred_F=model_best_F.predict(X_test_F)

# Print evaluation metrics

clr_F = classification_report(y_test, y_pred_F)


print("Classification Report SVM on F score data subset:\n----------------------\n", clr_F)
print( "Evaluation metrics for Kbest F score data subset")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_F))
print("Precision:",metrics.precision_score(y_test, y_pred_F, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_F, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_F, average = 'weighted'))


# ### Another approach 

# Below we have also tried another way: we trained the best SVC model on the F subset, and passed the same parameter range for gridsearch. The results are the same. The prediction score for Positive class falls by 1 percentage points, implying that in obtaining the F subset we are omitting some important features that could help predict class 2 better.

# In[172]:


svc_F = SVC(random_state=0).fit(X_train_F, y_train)
y_pred_F= svc_F.predict(X_test_F)
parameters_svc_F = {'kernel':['linear', 'rbf'], 
                  'C':[1, 0.5, 0.1, 10], 
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05, 0.5],
                  'class_weight': ['balanced']}

# Instantiating gridsearch object on the classifier
grid_search_svc_F = GridSearchCV(SVC(random_state=0), parameters_svc_F,cv=10, n_jobs=-1)
grid_search_svc_F.fit(X_train_F, y_train)
# Creating a model with best parameters
model_F=grid_search_svc_F.best_estimator_

# Applying model to test set to predict y
y_pred_F_best=model_F.predict(X_test_F)

# Print evaluation metrics
print(" Evaluation metrics for the best SVC model")
print("Classification Report SVM on F score data subset:\n----------------------\n", classification_report(y_test, y_pred_F_best))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_F_best))
print("Precision:",metrics.precision_score(y_test, y_pred_F_best, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_F_best, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_F_best, average = 'weighted'))


# ### 3.1.4. SVM on subset obtained through Mutual Information

# In[161]:


#Train the best hyperparameter model on the best F score training sets.
grid_search_svc.fit(X_train_MI, y_train)

# Creating a model with best parameters
model_best_MI=grid_search_svc.best_estimator_

# Applying model to get predicted y
y_pred_MI=model_best_MI.predict(X_test_MI)

# Print evaluation metrics

clr_MI = classification_report(y_test, y_pred_MI)
print("Classification Report SVM on MI data subset:\n----------------------\n", clr_MI)
print( "Evaluation metrics for Kbest MI data subset")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_MI))
print("Precision:",metrics.precision_score(y_test, y_pred_MI, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_MI, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_MI, average = 'weighted'))


# Below is the second way of training an SVC model on MI subset.
# Just like with the F subset, we train a SVC model on it and then do a gridsearch on the trained model, instead of applying the gridsearch parameters obtained from the training on the whole dataset.
#  

# In[170]:


clf = SVC(random_state=0).fit(X_train_MI, y_train)
y_pred= clf.predict(X_test_MI)
parameters_svc = {'kernel':['linear', 'rbf'], 
                  'C':[1, 0.5, 0.1, 10], 
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05, 0.5],
                  'class_weight': ['balanced']}

# Instantiating gridsearch object on the classifier
grid_search_svc_MI = GridSearchCV(SVC(random_state=0), parameters_svc, cv=10, n_jobs=-1)
grid_search_svc_MI.fit(X_train_MI, y_train)
# Creating a model with best parameters
model_best_MI=grid_search_svc_MI.best_estimator_

# Applying model to test set to predict y
y_pred_grs_MI=model_best_MI.predict(X_test_MI)

# Print evaluation metrics
print(" Evaluation metrics for the best SVC model")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_grs_MI))
print("Precision:",metrics.precision_score(y_test, y_pred_grs_MI, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_grs_MI, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_grs_MI, average = 'weighted'))


# In[164]:


print("\n The best parameters across ALL searched params:\n",grid_search_svc_MI.best_params_)


# ### Conclusion from Section 3.1
# 
# We used both LinearSVC and SVC models to train our data. 
# 1. We used LinearSVC because it works better with larger datasets. However, we did not find any advantage in that. Instead, the model was not converging after the default 1000 iterations and we had to increase number of iterations continuously. 
# The best parameters across ALL searched parameters for LinearSVC were:
# {'C': 0.1, 'max_iter': 8000}. In the end LinearSVC model achieved 95% accuracy and F1-score.
# It is worth noting, that LinearSVC and hyperparameter tuning was taking quite a long time to perform on the original dataset.
# 
# 2. Next, we applied SVC() model on our original dataset. It was taking too long to train the model, as well as get the hyperparameters. However, we achieved higher performance with this model. After training the hyperparameters, we achieved the following best parameters:
#  {'C': 10, 'class_weight': 'balanced', 'gamma': 0.0001, 'kernel': 'rbf'}.
# The F1 score is 96.1%. Because this model achieved higher results than the best performing LinearSVC model, we fitted this model on F and MI subsets. The F1 score from F subset is 95.9 %, and from MI subset 97.2%. 
#  
# 3. The results show that MI is an effective method to select a subset of relevant features for SVC algorithm. It not only took less to perform model fitting, but also showed an increase of 1.1 percentage points in the F1 score and accuracy compared to the best SVC model trained on the whole dataset.
# 
# 4. We have used two approached for training models on subsets:
# We obtained the best model based on the hyperparameters trained on the original dataset. Then we applied the best model on the two subsets.
# The other approach was - we tuned hyperparameters on the subsets and obtained the best models based on those individual subsets. 
# The evaluation metrics were exactly the same.

# ## Section 3.2. Random Forest Classifier <a class="anchor" id="section_3_2"></a>

# Random Forest is a supervised learning algorithm . The forest which it builds is an ensemble of a decision trees.
# Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.

# In[183]:


# Import RF classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score #Cross validation
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, roc_auc_score


# ### 3.2.1. RF on original dataset

# In[209]:


#Make predictions for the test set

# Train the modeL
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

y_pred_RF = forest.predict(X_test)

# View accuracy score
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_RF))

# confusion matrix for test data and predictions
confusion_matrix(y_test,y_pred_RF)


# In[284]:


# For negative emotions        For Neutral Emotions        For Positive Emotions
# True Positive = 189          True Positive = 213         True Positive = 226
# False Negative = 3           False Negative = 0          False Negative = 9
# False Positive = 9           False Positive = 0          False Positive = 3


# In[167]:


#Getting mean of accuracies which we will get from 10 folds.
np.mean(cross_val_score(forest,X_train,y_train,cv=10,scoring='accuracy'))


# ### Hyperparameter tuning on RF

# In[211]:


rfc=RandomForestClassifier(random_state=42)


# In[210]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[213]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[217]:


print(" Results from Grid Search " )
print("\n The best score across ALL searched params:\n",CV_rfc.best_score_)
print("\n The best parameters across ALL searched params:\n",CV_rfc.best_params_)


# In[216]:


# forest_grid.cv_results_


# In[218]:


model_best_rfc= CV_rfc.best_estimator_


# In[226]:


model_best_rfc  # the best performing model has the following parameters


# In[222]:


y_pred_rfc = model_best_rfc.predict(X_test)  # fitting the best model on the testing set to predict y outbut


# In[223]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rfc))
print("Precision:",metrics.precision_score(y_test, y_pred_rfc, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_rfc, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_rfc, average = 'weighted'))

# So after applying hyperparamter tuning our accuracy is 98.28% which increased by 0.20% of our original accuracy.


# In[232]:


clr_rfc=classification_report(y_test,y_pred_rfc)
print("Classification Report SVM on F score data subset:\n----------------------\n", clr_rfc)


# In[237]:


model_best_rfc.feature_importances_


# In[246]:


model_best_rfc.estimators_[5]
d = {'Stats':X.columns,'FI':model_best_rfc.feature_importances_}
cf=pd.DataFrame(d)
cf.sort_values('FI', ascending=False)


# In[259]:


import plotly.express as px
fig = px.bar_polar(df[:30], r="FI", theta="Stats",
                   color="Stats", template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()

#Code source: https://towardsdatascience.com/present-the-feature-importance-of-the-random-forest-classifier-99bb042be4cc


# ### 3.2.2. RF on "F" dataset  <a class="anchor" id="section_3_2_2"></a>

# In[227]:


#Train the best hyperparameter model on the best F score training sets.
CV_rfc.fit(X_train_F, y_train)

# Creating a model with best parameters
model_rfc_F=CV_rfc.best_estimator_

# Applying model to get predicted y
y_pred_rfc_F=model_rfc_F.predict(X_test_F)

# Print evaluation metrics

clr_rfc_F = classification_report(y_test, y_pred_rfc_F)


print("Classification Report SVM on F score data subset:\n----------------------\n", clr_rfc_F)
print( "Evaluation metrics for Kbest F score data subset")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rfc_F))
print("Precision:",metrics.precision_score(y_test, y_pred_rfc_F, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_rfc_F, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_rfc_F, average = 'weighted'))


# ### 3.2.3. RF on MI dataset <a class="anchor" id="section_3_2_3"></a>

# In[233]:


#Train the best hyperparameter model on the MI score training sets.
CV_rfc.fit(X_train_MI, y_train)

# Creating a model with best parameters
model_rfc_MI=CV_rfc.best_estimator_

# Applying model to get predicted y
y_pred_rfc_MI=model_rfc_MI.predict(X_test_MI)

# Print evaluation metrics

clr_rfc_MI = classification_report(y_test, y_pred_rfc_MI)


print("Classification Report SVM on MI score data subset:\n----------------------\n", clr_rfc_MI)
print( "Evaluation metrics for MI score data subset")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rfc_MI))
print("Precision:",metrics.precision_score(y_test, y_pred_rfc_MI, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred_rfc_MI, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred_rfc_MI, average = 'weighted'))


# **Wrapping up the results from the Random Forest Classifier applied to the three datasets**
# 
# 
# The best model performance was achieved from the Mutual Information dataset. The accuracy was 96.11% which was better than other two datasets. There was not a big difference between the accuracies. The original dataset had 96.04% whereas F_calssif dataset had 94.90. So we can say that the best peformance was on Mutual Information dataset.

# ## Section 3.3. Multi-layer Perceptron (MLP) <a class="anchor" id="section_3_3"></a>

# We start building one multi-layer perceptron by using the Keras Sequential model. It is a linear stack of layers.
# We are going to create MLP *Sequential* model by passing a list of layer-specific data to the constructor. Since we have more than two classes, activation function on our final layer can't be sigmoid, like it is in binary classification models. We will use *softmax* function to have 3-class output.

# In[353]:


# pip install pydot


# In[354]:


from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential, Input, Model 
from tensorflow.keras.utils import to_categorical  
# to convert y into categorical var
from keras.layers import Dense, Dropout 
from tensorflow.keras.utils import plot_model  # to visualize the model
from numpy import loadtxt
from tensorflow.keras.optimizers import RMSprop


# Step 1. Building the Neural Network with two hidden layers, considering that we have a large dataset.We will use dropout technique to prevent the model from overfitting.

# In[371]:


# Initialize the constructor
model = Sequential()


# In[372]:


# Add an input layer 
model.add(Dense(32, activation='relu', input_dim=2548))

# Add the first hidden layer 
model.add(Dense(16, activation='relu'))
          
# Add a dropout layer
model.add(Dropout(0.2)) # passing 20% of dropout rate

# Add the second hidden layer
model.add(Dense(8, activation='relu'))
# Add another dropout layer
model.add(Dropout(0.2))

# # Add the third hidden layer
# model.add(Dense(4, activation='relu'))
# # Add another dropout layer
# model.add(Dropout(0.2))

# Add an output layer 
model.add(Dense(3, activation="softmax")) # In this last layer each node contains a score indicating one of three classes.


# Step 2. Visualize the model architecuture

# In[373]:


plot_model(model,show_shapes=True, show_layer_names=True)

# Source:https://www.youtube.com/watch?v=iajq0xQZ2cQ


# Step 2. Compile the model

# In[423]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Source: read://https_faroit.com/?url=https%3A%2F%2Ffaroit.com%2Fkeras-docs%2F1.0.1%2Fgetting-started%2Fsequential-model-guide%2F


# In[375]:


model.summary()


# Step 3. Train the model

# In[376]:


y_train_mlp = to_categorical(y_train, 3)

y_test_mlp=to_categorical(y_test,3)
y_train_mlp


# In[424]:


history=model.fit(X_train, y_train_mlp, validation_data=(X_test, y_test_mlp), shuffle=True, epochs=50, batch_size=32)


# In[380]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

#Source : tutorial week 11 code


# In[381]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[385]:


y_pred = np.argmax(model.predict(X_test), axis=-1)
#As the predict_classes didn't work, source: https://stackoverflow.com/questions/68883510/attributeerror-sequential-object-has-no-attribute-predict-classes


# In[67]:


# print(y_pred)
# print(y_test)
# print(X_test)
# #print(y_test.shape)
#print(y_test[:10])
# print(pd.DataFrame(y_pred[:10])) # comparing the results by eye
# print(pd.DataFrame(y_test[:10])) 


# In[142]:


#score = model.evaluate(X_test, y_test,verbose=0)


# In[425]:


loss,accuracy=model.evaluate(X_test,y_test_mlp,verbose=1)
print("Model Loss: %.2f, Accuracy: %.2f" % ((loss), (accuracy*100)))


# Changing hidden layers and number of nodes.

# In[426]:


# Initialize the constructor
mode = Sequential()


# In[450]:


model = Sequential()
# Add an input layer 
model.add(Dense(32, activation='relu', input_dim=2548))

# Add the first hidden layer 
model.add(Dense(16, activation='relu'))
          
# Add a dropout layer
model.add(Dropout(0.2)) # passing 20% of dropout rate

# Add the second hidden layer
model.add(Dense(8, activation='relu'))
# Add another dropout layer
model.add(Dropout(0.2))

# # Add the third hidden layer
# model.add(Dense(4, activation='relu'))
# # Add another dropout layer
# model.add(Dropout(0.2))

# Add an output layer 
model.add(Dense(3, activation="softmax")) # In this last layer each node contains a score indicating one of three classes.


# In[451]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[461]:


history1=model.fit(X_train, y_train_mlp,validation_data=(X_test, y_test_mlp), shuffle=True, epochs=130, batch_size=32)


# Because the number of epochs was showing to be producing better in terms of accuracy, was around 10.

# In[462]:


loss,accuracy=model.evaluate(X_test,y_test_mlp,verbose=1)
print("Model Loss: %.2f, Accuracy: %.2f" % ((loss), (accuracy*100)))


# Make a prediciton

# In[463]:


y_pred1=np.argmax(model.predict(X_test), axis=-1)


# In[464]:


from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
print("Precision:",metrics.precision_score(y_test, y_pred1, average='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred1, average='weighted'))


# In[177]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm


# ### 3.3.1. Tuning of hyperparameters
# 
# For this task, we have used the object MLPClassifier, as it is more consistent with the ways of tuning hyperparameters for other models. However, we had to interrupt kernel because of the concerns that it was taking more than two hours on the laptop. 
# We still have provided the approach to hyperparameter tuning below, based on MLPClassifier object.
# One of the drawbacks of this method is that the solver method "rmsprop" used for multiclass classification tasks is not supported by this object.

# In[263]:


from sklearn.neural_network import MLPClassifier
# Call the MLPClassifier object with its default parameters.
mlp=MLPClassifier(max_iter=500, activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=42, shuffle=True, solver='lbfgs',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)

# Fit the model to training input
mlp.fit(X_train, y_train)
# Make prediction based on test set
pred = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
# Generate confusion matrix and print it
confusion_matrix(y_test,pred)

print(classification_report(y_test,pred))


# In[262]:


parameter_space = {
   'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
   'activation': ['tanh', 'relu'],
   'solver': ['sgd', 'rmsprop'],
   'alpha': [0.0001, 0.05],
   'learning_rate': ['constant','adaptive'],
}


grid = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
grid.fit(X_train, y_train)


# Best parameter set
print('Best parameters found:\n', grid.best_params_)

y_pred = grid.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

