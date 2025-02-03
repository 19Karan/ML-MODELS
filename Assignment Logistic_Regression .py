#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_train=pd.read_csv("Titanic_train.csv")


# In[3]:


data_train.head()


# In[4]:


data_test=pd.read_csv("Titanic_test.csv")


# In[5]:


data_test.head()


# In[6]:


data_train.info()


# In[7]:


data_train.isnull().sum()


# In[8]:


age_mean=data_train['Age'].mean()


# In[64]:


data_train['Age'].fillna(age_mean,inplace=True)


# In[10]:


cabin_mode=data_train['Cabin'].mode()


# In[11]:


cabin_mode


# In[12]:


# data_train['Cabin'].fillna(cabin_mode,inplace=True)


# In[13]:


data_train.isnull().sum()


# In[14]:


# data_train['Cabin'].fillna(0)


# In[15]:


# data_train['Cabin'].drop


# In[16]:


data_train


# In[17]:


data_train.describe()  #here we can see the statistics of data


# In[18]:


data_train.isnull().sum()


# In[19]:


# data_train.drop(['Cabin'],axis=1,inplace=True)  #here we drop unecessary columns


# In[20]:


data_train


# In[21]:


data_train.dtypes


# In[22]:


em=data_train["Embarked"].mode()


# In[23]:


data_train['Embarked'].fillna(em)


# In[24]:


#we done with eda process now we do visulization
# Histograms for numerical features
data_train.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Features", fontsize=16)
plt.tight_layout()
plt.show()


# In[25]:


# # Pair plot for relationships
# sns.pairplot(data_train, hue='Survived')  
# plt.suptitle("Pair Plot of Features", y=1.02, fontsize=16)
# plt.show()


# In[26]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_train.select_dtypes(include=['int','float']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


# ### Data Preprocessing
# Handle Missing Values and Encode Categorical Variables

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   #here we import important packages


# In[28]:


#split data into x and y
X=data_train.drop(columns=['Survived'])
y=data_train['Survived']


# In[29]:


X


# In[30]:


# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X.select_dtypes(include='int'))


# In[31]:


X


# In[32]:


# Split into training and testing sets (80% training, 20% testing)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# ###  Model Building
# Train Logistic Regression Model

# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


model=LogisticRegression()


# In[35]:


model


# In[36]:


model.fit(X_train,y_train)


# ### Model Evaluation
# Evaluate Model Performance

# In[37]:


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)


# In[38]:


y_pred_train=model.predict(X_train)  #here we do prediction on seen data


# In[39]:


y_pred_train


# In[40]:


# Predictions
y_pred_test = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]


# In[41]:


y_pred_test
y_pred_prob


# In[42]:


# Metrics see the model perfomance using the matrics
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_pred_prob)


# In[43]:


accuracy,precision,recall,f1,roc_auc


# In[ ]:





# In[44]:


accuracy = accuracy_score(y_train,y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)
# roc_auc = roc_auc_score(y_test, y_pred_prob)


# In[45]:


accuracy,precision,recall,f1  


# In[46]:


#we can see the accuracy of traning and testing data model give accuracy by looking accuracy we can say model is underfiting


# In[47]:



# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()


# ### Interpretation of Coefficients

# In[53]:


# Coefficients of the logistic regression model
# coefficients = pd.DataFrame({
#     "Feature": data_train.drop(columns=['Survived']).columns,  # Replace 'target' with your target column
#     "Coefficient": model.coef_[0]
# })
# coefficients["Importance"] = coefficients["Coefficient"].abs()
# coefficients.sort_values(by="Importance", ascending=False, inplace=True)

# print("Feature Importance:")
# print(coefficients)

# Logistic Regression Coefficients

print("\nLogistic Regression Coefficients:")
for feature, coef in zip(data_train.columns, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")


# In[ ]:


#we are not done with model deployment


# In[58]:


# ! pip install streamlit


# In[67]:


# import streamlit as st
# import numpy as np

# st.title("Titanic Survival Predictor")

# st.write("Input the passenger details to predict survival")

# user_input = []
# for feature in data_train.columns:
#     user_input.append(st.number_input(f"Enter value for {feature}:"))

# if st.button("Predict"):
#     user_input = np.array(user_input).reshape(1, -1)
#     user_input_scaled = scaler.transform(user_input)
#     prediction = logistic_model.predict(user_input_scaled)[0]
#     prediction_proba = logistic_model.predict_proba(user_input_scaled)[0][1]

#     st.write(f"Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
#     st.write(f"Probability of Survival: {prediction_proba:.2f}")


# ### Interview Questions
# 
# 
# 1. What is the Difference Between Precision and Recall?
# 
# Precision: The proportion of true positive predictions out of all positive predictions.
#  
# Recall (Sensitivity): The proportion of true positive predictions out of all actual positive cases.
#  
# Trade-off: Higher precision reduces false positives, while higher recall reduces false negatives.
# 
# 
# 2. What is Cross-Validation, and Why Is It Important in Binary Classification?
# 
# Definition: Cross-validation splits the dataset into multiple training and validation sets to evaluate the model's performance.
# Why Important?
# Reduces overfitting by testing on unseen data.
# Provides a more reliable estimate of model performance.
# Ensures the model generalizes well.
# 
# 

# In[1]:


jupyter nbconvert --to script your_notebook.ipynb

