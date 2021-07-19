########################### problem1 #############################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
Affairs_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Logistic Regression/Affairs.csv", sep = ",")
Affairs_data.columns

Affairs_data['naffairs'].describe()
Affairs_data.dtypes
#EDA
#removing Unnamed: 0
Affairs_data = Affairs_data.drop('Unnamed: 0', axis = 1)
Affairs_data.head(11)
Affairs_data.describe()
Affairs_data.isna().sum()

#covering continuous data to discrete data of naffairs
Affairs_data['naffairs'] = np.where(Affairs_data['naffairs'] >6, 1 , 0)
Affairs_data.dtypes

# Model building 
# import statsmodels.formula.api as sm
Affairs_model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = Affairs_data).fit()

#summary
Affairs_model.summary() # for AIC
pred = Affairs_model.predict(Affairs_data.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(Affairs_data.naffairs, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
Affairs_data["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
Affairs_data.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(Affairs_data["pred"], Affairs_data["naffairs"])
classification

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Affairs_data, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()

#summary
model.summary() # for AIC

# Prediction on Test data set
test_pred = model.predict(test_data.iloc[:,1:])

# Creating new column for storing predicted class of naffairs
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(test_data['naffairs'],test_data.test_pred)
accuracy_test
# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = accuracy_score(train_data['naffairs'],train_data.train_pred)
accuracy_test

############################## problem2 ###########################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
Advertising_data = pd.read_csv("C:/Users/DELL/Downloads/advertising.csv", sep = ",")
Advertising_data.columns
print(Advertising_data.describe())
Advertising_data.dtypes

#EDA
Advertising_data = Advertising_data.iloc[:,[0,1,2,3,6,9]]
Advertising_data.columns

#renaming the column
Advertising_data.rename(columns = {'Daily_Time_ Spent _on_Site': 'Daily_Time_Spent_on_Site','Daily Internet Usage' : 'Daily_Internet_Usage'},inplace = True)

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male', data = Advertising_data).fit()
#summary
logit_model.summary() # for AIC

pred = logit_model.predict(Advertising_data.iloc[ :, 0:5])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(Advertising_data.Clicked_on_Ad, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
Advertising_data["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
Advertising_data.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(Advertising_data["pred"], Advertising_data["Clicked_on_Ad"])
classification

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Advertising_data, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male', data = Advertising_data).fit()

#summary
model.summary() # for AIC

# Prediction on Test data set
test_pred = model.predict(test_data.iloc[:,:5])

# Creating new column for storing predicted class of naffairs
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(test_data['Clicked_on_Ad'],test_data.test_pred)
accuracy_test
# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :5 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = accuracy_score(train_data['Clicked_on_Ad'],train_data.train_pred)
accuracy_test

####################################### problem3 ###################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
Election_data = pd.read_csv("C:/Users/DELL/Downloads/election_data.csv", sep = ",")
Election_data.columns
print(Election_data.describe())
Election_data.dtypes

#EDA
Election_data = Election_data.iloc[:,1:]
#drop NA values
Election_data = Election_data.dropna()

#renaming the column
Election_data.rename(columns = {'Amount Spent': 'Amount_Spent','Popularity Rank' : 'Popularity_Rank'},inplace = True)
Election_data.columns
# Model building 
# import statsmodels.formula.api as sm
logit = sm.logit('Result ~ Year + Amount_Spent + Popularity_Rank ', data = Election_data)
logit_model= logit.fit(method = 'bfgs')
#summary
logit_model.summary() # for AIC

pred = logit_model.predict(Election_data.iloc[ :, 1:])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(Election_data.Result, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
Election_data["pred"] = np.zeros(10)
# taking threshold value and above the prob value will be treated as correct value 
Election_data.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(Election_data["pred"], Election_data["Result"])
classification

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(Election_data, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Result ~ Year + Amount_Spent + Popularity_Rank ', data = Election_data)
logit_model= logit.fit(method = 'bfgs')
#summary
logit_model.summary() # for AIC

# Prediction on Test data set
test_pred = logit_model.predict(test_data.iloc[:,1:])

# Creating new column for storing predicted class of naffairs
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(3)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix

from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(test_data['Result'],test_data.test_pred)
accuracy_test
# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Result"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = logit_model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(7)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Result'])
confusion_matrx

accuracy_train = accuracy_score(train_data['Result'],train_data.train_pred)
accuracy_test




