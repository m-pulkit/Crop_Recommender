import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score, \
    confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle


data = pd.read_excel('Crop_recommendation.xlsx')
print(data.head(15))
print(data.describe())

print(data.info())
print([x for x in data.N if type(x) != int])

data.replace("?", np.nan, inplace=True)
data.replace(" ?",np.nan,inplace=True)
data.replace("? ",np.nan,inplace=True)

# data = data.convert_dtypes()
print(data.info())

print('Null Values in dataset:\n', data.isnull().sum(), sep='')
print(data.describe())
print(data.describe(include='O'))
print('No. of Zeros in column \'N', (data['N'] == 0).sum())

data['N'].replace(0, data.loc[data['N'] != 0, 'N'].mean(), inplace=True)
data['P'].replace(0, data.loc[data['P'] != 0, 'P'].mean(), inplace=True)
data['K'].replace(0, data.loc[data['K'] != 0, 'K'].mean(), inplace=True)
data['temperature'].replace(0, data.loc[data['temperature'] != 0, 'temperature'].mean(), inplace=True)
data['humidity'].replace(0, data.loc[data['humidity'] != 0, 'humidity'].mean(), inplace=True)
data['ph'].replace(0, data.loc[data['ph'] != 0, 'ph'].mean(), inplace=True)

data.fillna(data['N'].mean(), inplace=True)
data.fillna(data['P'].mean(), inplace=True)
data.fillna(data['K'].mean(), inplace=True)
data.fillna(data['temperature'].mean(), inplace=True)
data.fillna(data['humidity'].mean(), inplace=True)
data.fillna(data['ph'].mean(), inplace=True)

print('Null Values in Data:\n', data.isnull().sum(), sep='')

plt.figure(figsize=(25, 20))
sns.barplot(x="label", y="N", data=data)
plt.xticks(rotation=90)
plt.rc('xtick', labelsize=6)
plt.xlabel('Crop')
plt.title('Amount of Nitrogen For various crops', fontsize=24)
plt.savefig('Plots/Nitrogen per Crop.jpg')

plt.figure(figsize=(25, 20))
sns.barplot(x="label", y="P", data=data)
plt.xticks(rotation=90)
plt.rc('xtick', labelsize=6)
plt.xlabel('Crop')
plt.title('Amount of Phosphorous For various crops', fontsize=24)
plt.savefig('Plots/Phosphorus per Crop.jpg')

sns.pairplot(data=data, hue='label')
plt.suptitle('Pair Plot for the dataset', fontsize=24)
plt.savefig('Plots/Pair Plot.jpg')
# plt.show()

plt.figure(figsize=(8, 10))
sns.displot(data["P"], color='r', kde=True)
plt.title('Histplot for Amount of Phosphorous')
plt.savefig('Plots/Phosphorus Distribution.jpg', bbox_inches='tight')

print('Duplicates in data', data.duplicated().sum())

b = data[["N", "K", "P", "humidity", "ph", "rainfall", "temperature"]]

plt.figure(figsize=(10, 6))
plotnumber = 1

for columns in b:
    if plotnumber <= 7:
        ax = plt.subplot(4, 3, plotnumber)
        sns.boxplot(b[columns])
        plt.xlabel(columns, fontsize=10)

    plotnumber += 1

plt.suptitle('Box Plots of all Features', fontsize=16)
plt.tight_layout()
plt.savefig('Plots/Box Plots.jpg')


## Construction and Evaluation of Models
Lc = LabelEncoder()
data.label = Lc.fit_transform(data['label'])

Crop_Mappings = dict(zip(Lc.classes_, Lc.transform(Lc.classes_)))
print('Crop_Mappings: \n', Crop_Mappings, sep='')
print('Dataset:\n', data.head(), sep='')

tc = data.corr()
print('Data Correlation:\n', tc, sep='')

plt.figure(figsize=(20, 18))
sns.heatmap(tc, annot=True, square=True, fmt='.2f', cmap='coolwarm',
            linewidths=1, annot_kws={'fontsize': 13, 'fontstyle': 'italic', 'fontweight': 'bold'},
            linecolor='black')

# annot_kws is used to change the text properties, typically the font size ,
# fmt(format) means adding text to each on each cell and .2f is used for placement of 2 digits after decimal
plt.title('Correlation b/w Features', fontsize=24)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.savefig('Plots/Correlation of Features.jpg')

x = data.drop('label', axis=1)
print('X:\n', x, sep='')
y = data['label']
print('Y:\n', y, sep='')


# # Model Building
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


## Logistic Regression
Log_Reg = LogisticRegression(max_iter=200)
Log_Reg.fit(x_train, y_train)

# **Evaluation of train data**
y_train_pred_LR = Log_Reg.predict(x_train)
print('Predictions: \n', y_train_pred_LR[:10], sep='')

Log_Reg_Train_Accuracy = accuracy_score(y_train, y_train_pred_LR)
print('Log Reg Train Accuracy:', Log_Reg_Train_Accuracy)

Log_Reg_Train_F1 = f1_score(y_train, y_train_pred_LR, average='weighted')
print('Log Reg Train F1 score:', Log_Reg_Train_F1)

Log_Reg_Train_Precision = precision_score(y_train, y_train_pred_LR, average='weighted')
print('Log Reg Train Precision score:', Log_Reg_Train_Precision)

Log_Reg_Train_Recall = recall_score(y_train, y_train_pred_LR, average='weighted')
print('Log Reg Train Recall score:', Log_Reg_Train_Recall)


# **Evaluation of test data**
y_pred_LR = Log_Reg.predict(x_test)
print(y_pred_LR[:10])

Log_Reg_Test_Accuracy = accuracy_score(y_test, y_pred_LR)
print('Log Reg Test Accuracy:', Log_Reg_Test_Accuracy)

Log_Reg_Test_F1 = f1_score(y_test, y_pred_LR, average='weighted')
print('Log Reg Test F1 score:', Log_Reg_Test_F1)

Log_Reg_Test_Precision = precision_score(y_test, y_pred_LR, average='weighted')
print('Log Reg Test Precision score:', Log_Reg_Test_Precision)

Log_Reg_Test_Recall = recall_score(y_test, y_pred_LR, average='weighted')
print('Log Reg Test Recall score:', Log_Reg_Test_Recall)

Log_Reg_cm = confusion_matrix(y_test, y_pred_LR)
print("Confusion Matrix:\n", Log_Reg_cm, sep='')

report = classification_report(y_test, y_pred_LR)
print("Classification Report:\n", report, sep='')


## Support Vector Machine
SVM_class = SVC(C=12.0, kernel='poly', gamma='scale')
SVM_class.fit(x_train, y_train)

# **Evaluation of train data**
y_train_pred_SVM = SVM_class.predict(x_train)
print(y_train_pred_SVM[:10])

SVM_Train_Accuracy = accuracy_score(y_train, y_train_pred_SVM )
print ('SVM Test Accuracy:', SVM_Train_Accuracy)

SVM_Train_F1 = f1_score(y_train, y_train_pred_SVM, average = 'weighted')
print ('SVM Test F1 score:', SVM_Train_F1)

SVM_Train_Precision = precision_score(y_train, y_train_pred_SVM, average = 'weighted')
print ('SVM Test Precision score:', SVM_Train_Precision)

SVM_Train_Recall = recall_score(y_train, y_train_pred_SVM, average = 'weighted')
print ('SVM Test Recall score:', SVM_Train_Recall)

# **Evaluation of Test Data**
y_pred_SVM = SVM_class.predict(x_test)
print('Predictions: \n', y_pred_SVM[:10], sep='')

SVM_Test_Accuracy = accuracy_score(y_test, y_pred_SVM)
print('SVM Test Accuracy:', SVM_Test_Accuracy)

SVM_Test_F1 = f1_score(y_test, y_pred_SVM, average='weighted')
print('SVM Test F1 score:', SVM_Test_F1)

SVM_Test_Precision = precision_score(y_test, y_pred_SVM, average='weighted')
print('SVM Test Precision score:', SVM_Test_Precision)

SVM_Test_Recall = recall_score(y_test, y_pred_SVM, average='weighted')
print('SVM Test Recall score:', SVM_Test_Recall)

SVM_cm = confusion_matrix(y_test, y_pred_SVM)
print('Confusion Matrix: \n', SVM_cm, sep='')

report = classification_report(y_test, y_pred_SVM)
print("Report:\n", report, sep='')


### Decision Tree Classifier
DT = DecisionTreeClassifier(criterion='gini', min_samples_split=2,
                            min_samples_leaf=1, random_state=2, splitter='random')
DT.fit(x_train, y_train)

# **Evaluation of train data**
y_train_pred_DT = DT.predict(x_train)
print('Predictions: \n', y_train_pred_DT[:10], sep='')

DT_Train_Accuracy = accuracy_score(y_train, y_train_pred_DT)
print('DT Train Accuracy:', DT_Train_Accuracy)

DT_Train_F1 = f1_score(y_train, y_train_pred_DT, average='weighted')
print('DT Train F1 score:', DT_Train_F1)

DT_Train_Precision = precision_score(y_train, y_train_pred_DT, average='weighted')
print('DT Train Precision score:', DT_Train_Precision)

DT_Train_Recall = recall_score(y_train, y_train_pred_DT, average='weighted')
print('DT Train Recall score:', DT_Train_Recall)

# **Evaluation of Test data**
y_pred_DT = DT.predict(x_test)
print('Predictions: \n', y_pred_DT[:10], sep='')

DT_Test_Accuracy = accuracy_score(y_test, y_pred_DT)
print('DT Test Accuracy:', DT_Test_Accuracy)

DT_Test_F1 = f1_score(y_test, y_pred_DT, average='weighted')
print('DT Test F1 score:', DT_Test_F1)

DT_Test_Precision = precision_score(y_test, y_pred_DT, average='weighted')
print('DT Test Precision score:', DT_Test_Precision)

DT_Test_Recall = recall_score(y_test, y_pred_DT, average='weighted')
print('DT Test Recall score:', DT_Test_Recall)

DT_report = classification_report(y_test, y_pred_DT)
print("Report:\n", DT_report, sep='')

DT_cm = confusion_matrix(y_test, y_pred_DT)
print('Confusion Matrix:\n: \n', DT_cm, sep='')


### Random Forest Classifier
RF_classifier = RandomForestClassifier(n_estimators=900, criterion='gini', random_state=0)
RF_classifier.fit(x_train, y_train)

# **Evaluation of Train data**
y_train_pred_RF = RF_classifier.predict(x_train)
print('Predictions: \n', y_train_pred_RF[:10], sep='')

RF_Train_Accuracy = accuracy_score(y_train, y_train_pred_RF)
print('RF Train Accuracy:', RF_Train_Accuracy)

RF_Train_F1 = f1_score(y_train, y_train_pred_RF, average='weighted')
print('RF Train F1 score:', RF_Train_F1)

RF_Train_Precision = precision_score(y_train, y_train_pred_RF, average='weighted')
print('RF Train Precision score:', RF_Train_Precision)

RF_Train_Recall = recall_score(y_train, y_train_pred_RF, average='weighted')
print('RF Train Recall score:', RF_Train_Recall)

# **Evaluation of Test data**
y_pred_RF = RF_classifier.predict(x_test)
print('Predictions: \n', y_pred_RF[:10], sep='')

RF_Test_Accuracy = accuracy_score(y_test, y_pred_RF)
print('RF Test Accuracy:', RF_Test_Accuracy)

RF_Test_F1 = f1_score(y_test, y_pred_RF, average='weighted')
print('RF Test F1 score:', RF_Test_F1)

RF_Test_Precision = precision_score(y_test, y_pred_RF, average='weighted')
print('RF Test Precision score:', RF_Test_Precision)

RF_Test_Recall = recall_score(y_test, y_pred_RF, average='weighted')
print('RF Test Recall score:', RF_Test_Recall)

RF_cm = confusion_matrix(y_test, y_pred_RF)
print('Predictions: \n', RF_cm, sep='')

RF_report = classification_report(y_test, y_pred_RF)
print('Report:\n', RF_report, sep='')


### Gradient Boosting Classifier
GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)

# **Evaluation of Train data**
y_train_pred_GBC = GBC.predict(x_train)
print('Predictions: \n', y_train_pred_GBC[:10], sep='')

GBC_Train_Accuracy = accuracy_score(y_train, y_train_pred_GBC)
print('GBC Train Accuracy:', GBC_Train_Accuracy)

GBC_Train_F1 = f1_score(y_train, y_train_pred_GBC, average='weighted')
print('GBC Train F1 score:', GBC_Train_F1)

GBC_Train_Precision = precision_score(y_train, y_train_pred_GBC, average='weighted')
print('GBC Train Precision score:', GBC_Train_Precision)

GBC_Train_Recall = recall_score(y_train, y_train_pred_GBC, average='weighted')
print('GBC Train Recall score:', GBC_Train_Recall)

# **Evaluation of Test data**
y_pred_GBC = GBC.predict(x_test)
print('Predictions: \n', y_pred_GBC[:10], sep='')

GBC_Test_Accuracy = accuracy_score(y_test, y_pred_GBC)
print('GBC Test Accuracy:', GBC_Test_Accuracy)

GBC_Test_F1 = f1_score(y_test, y_pred_GBC, average='weighted')
print('GBC Test F1 score:', GBC_Test_F1)

GBC_Test_Precision = precision_score(y_test, y_pred_GBC, average='weighted')
print('GBC Test Precision score:', GBC_Test_Precision)

GBC_Test_Recall = recall_score(y_test, y_pred_GBC, average='weighted')
print('GBC Test Recall score:', GBC_Test_Recall)

GBC_cm = confusion_matrix(y_test, y_pred_GBC)
print('Confusion Matrix \n', GBC_cm, sep='')

GBC_report = classification_report(y_test, y_pred_GBC)
print('Report:\n', GBC_report, sep='')


### XGBoost Classifier
xg = XGBClassifier(n_estimators=120, max_depth=5, n_jobs=-1, random_state=5)
xg.fit(x_train, y_train)

# **Evaluation of Train data**
y_train_pred_xg = xg.predict(x_train)
print('Predictions: \n', y_train_pred_xg[:10], sep='')

Xg_Train_Accuracy = accuracy_score(y_train, y_train_pred_xg)
print('Xg Train Accuracy:', Xg_Train_Accuracy)

Xg_Train_F1 = f1_score(y_train, y_train_pred_xg, average='weighted')
print('Xg Train F1 score:', Xg_Train_F1)

Xg_Train_Precision = precision_score(y_train, y_train_pred_xg, average='weighted')
print('Xg Train Precision score:', Xg_Train_Precision)

Xg_Train_Recall = recall_score(y_train, y_train_pred_xg, average='weighted')
print('Xg Train Recall score:', Xg_Train_Recall)

# **Evaluation of Test data**
y_pred_xg = xg.predict(x_test)
print('Predictions: \n', y_pred_xg[:10], sep='')

Xg_Test_Accuracy = accuracy_score(y_test, y_pred_xg)
print('Xg Test Accuracy:', Xg_Test_Accuracy)

Xg_Test_F1 = f1_score(y_test, y_pred_xg, average='weighted')
print('Xg Test F1 score:', Xg_Test_F1)

Xg_Test_Precision = precision_score(y_test, y_pred_xg, average='weighted')
print('Xg Test Precision score:', Xg_Test_Precision)

Xg_Test_Recall = recall_score(y_test, y_pred_xg, average='weighted')
print('Xg Test Recall score:', Xg_Test_Recall)

xg_cm = confusion_matrix(y_test, y_pred_xg)
print('Confusion Matrix: \n', xg_cm, sep='')

xg_report = classification_report(y_test, y_pred_xg)
print('Report:\n', xg_report, sep='')


# ## Conclusion
results = pd.DataFrame({
    "Classifier": ["Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest", "Gradiet Boosting", "XGB"],
    "Precision": [Log_Reg_Test_Precision, SVM_Test_Precision, DT_Test_Precision, RF_Test_Precision, GBC_Test_Precision, Xg_Test_Precision],
    "Recall": [Log_Reg_Test_Recall, SVM_Test_Recall, DT_Test_Recall, RF_Test_Recall, GBC_Test_Recall, Xg_Test_Recall],
    "F1": [Log_Reg_Test_F1, SVM_Test_F1, DT_Test_F1, RF_Test_F1, GBC_Test_F1, Xg_Test_F1],
    "Accuracy": [Log_Reg_Test_Accuracy, SVM_Test_Accuracy, DT_Test_Accuracy, RF_Test_Accuracy, GBC_Test_Accuracy, Xg_Test_Accuracy]
})
print(results.round(2))

plt.figure(figsize=(12, 5))
ax = sns.barplot(results.melt('Classifier').rename(columns=str.title), x='Classifier', y='Value', hue='Variable')
plt.legend(loc="center")
ax.set_ybound(0.4)
plt.rc('xtick', labelsize=8)
plt.title("Result of all Classifiers", fontsize=14);
plt.savefig("Plots/All Classifiers' Result.png")

# We find that even though most model perform really well, the Random Forest Clasifier gives the highest scores and hence we select it as our prime model.
plt.figure(figsize=(10, 10))
sns.barplot(x=RF_classifier.feature_importances_, y=x.columns)
plt.title("Features' Importance in Random Forest Model", fontsize=14);
plt.ylabel('Features')
plt.savefig('Plots/Feature Importance of RF Classifier.png', bbox_inches='tight')
# plt.show()
# Thus we see that rainfall and humidity are two of the most influential factors in deciding which crop should be grown
# Let's finish the project by saving the model as a pickle file

with open('Crop_RF_classifier.pkl', 'wb') as pickle_out:
    pickle.dump(Crop_Mappings, pickle_out)
    pickle.dump(RF_classifier, pickle_out)
