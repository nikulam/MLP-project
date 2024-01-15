#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
#%%
data = pd.read_csv('C:/Users/mirom/Desktop/IST/ic/project/Proj1_Dataset.csv', delimiter=',')

#See if there are any missing values
print(data.isnull().sum())

#Drop rows with missing values
data.dropna(axis=0, inplace=True)

#%%
#Create 'timestamp column' with datetime objects
#Remove extra columns
temp = ['/' for i in range(0, data.shape[0])]
format = '%d/%m/%Y/%H:%M:%S'
data['daytime'] = data['Date'] + temp + data['Time']
data['timestamp'] = pd.to_datetime(data['daytime'], format=format)
data.drop(['Date', 'Time', 'daytime'], axis=1, inplace=True)

#%%
#Plot feature S1Temp to find outliers
plt.scatter(data['timestamp'], data['S1Temp'])
plt.xlabel('Date and Time')
plt.ylabel('S1Temp (°C)')
plt.title('S1 Temperature as a function of time')
plt.show()

#%%
#Use mean and standard deviation to verify and remove outliers
threshold = 4
outlier_columns = ['S1Temp', 'S3Temp', 'S1Light', 'S3Light']
for c in outlier_columns:
    mean1 = np.mean(data[c])
    std1 = np.std(data[c])
    outliers = []
    for n in data[c]:
            z_score= (n - mean1) / std1 
            if np.abs(z_score) > threshold:
                outliers.append(n)
    
    for o in outliers:
        print(c, o)
        data = data[data[c] != o]

#%%
#Smoothen column S3Light
#data['S3Light'] = data['S3Light'].rolling(25).mean()


#%%
#Plot correlation matrix as a heatmap to get intuition about 
#pairwise correlations of different features.
plot_data = data.drop(['timestamp'], axis=1)
plt.matshow(plot_data.corr())
plt.xticks(np.arange(len(plot_data.columns)), plot_data.columns, rotation=90)
plt.yticks(np.arange(len(plot_data.columns)), plot_data.columns, rotation=0)
plt.colorbar()
plt.show()


#%%
#Drop correlative features
data.drop(['S3Temp'], axis=1, inplace=True)


#%%
#Split data into train, val and test sets and train the model
X = data[['S1Temp', 'S2Temp', 'S1Light', 'S2Light', 'S3Light', 'CO2', 'PIR1', 'PIR2']]
y = data['Persons']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

#Normalize numeric features
for c in X_train.columns:
    train_mean = X_train[c].mean()
    train_std = X_train[c].std()
    X_train[c] = X_train[c].apply(lambda x: (x-train_mean) / train_std)
    X_val[c] = X_val[c].apply(lambda x: (x-train_mean) / train_std)
    X_test[c] = X_test[c].apply(lambda x: (x-train_mean) / train_std)


mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate_init=0.1)

mlp.fit(X_train, y_train)

#%%
#Make predictions and print metrics on training and validation data
y_train_pred = mlp.predict(X_train)
y_val_pred = mlp.predict(X_val)

print('training accuracy: ', accuracy_score(y_train, y_train_pred))
print('validation accuracy: ', accuracy_score(y_val, y_val_pred))
print('training precision: ', precision_score(y_train, y_train_pred, average='macro'))
print('validation precision: ', precision_score(y_val, y_val_pred, average='macro'))
print('training recall: ', recall_score(y_train, y_train_pred, average='macro'))
print('validation recall: ', recall_score(y_val, y_val_pred, average='macro'))
print('training matrix: ', confusion_matrix(y_train, y_train_pred))
print('validation matrix: ', confusion_matrix(y_val, y_val_pred))
print('training f1: ', f1_score(y_train, y_train_pred, average='macro'))
print('validation f1: ', f1_score(y_val, y_val_pred, average='macro'))


#%%
#Make predictions and print metrics on test data
y_test_pred = mlp.predict(X_test)

print('test accuracy: ', accuracy_score(y_test, y_test_pred))
print('test precision: ', precision_score(y_test, y_test_pred, average='macro'))
print('test recall: ', recall_score(y_test, y_test_pred, average='macro'))
print('test f1: ', f1_score(y_test, y_test_pred, average='macro'))

matrix = confusion_matrix(y_test, y_test_pred)
ax= plt.subplot()
sns.heatmap(matrix, annot=True, fmt='g', ax=ax, vmax=50)
ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
ax.set_title('Confusion Matrix',fontsize=15)

#%%
#Divide column persons: 0 if <= 2, 1 if > 2
#data['Persons'] = [0 if n <= 2 else 1 for n in data['Persons']]
binary_y = [0 if n <= 2 else 1 for n in y_test]
binary_y_pred = [0 if n <= 2 else 1 for n in y_test_pred]

#%%
#Metrics and matrix for binary classification
print('test accuracy: ', accuracy_score(binary_y, binary_y_pred))
print('test precision: ', precision_score(binary_y, binary_y_pred, average='macro'))
print('test recall: ', recall_score(binary_y, binary_y_pred, average='macro'))
print('test f1: ', f1_score(binary_y, binary_y_pred, average='macro'))

matrix = confusion_matrix(binary_y, binary_y_pred)
ax= plt.subplot()
sns.heatmap(matrix, annot=True, fmt='g', ax=ax, vmax=50)
ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
ax.set_title('Confusion Matrix',fontsize=15)

#%%
#Save the model in a file
'''
save = input('Do you want to save the model (type yes to confirm)? ').lower()
if save != 'yes':
    print('Model not saved.')
else:
    with open('mlp.pkl', 'wb') as file:
        pickle.dump(mlp, file)
    print('Model saved to mlp.pkl.')
'''

