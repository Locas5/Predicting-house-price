
# coding: utf-8

# # Part 1 Data Exploration

# ## In this section, we are going to conduct a very simple data exploration. The dataset we use is a data of houses in an area.

# In[1]:


import pandas as pd 


# In[2]:


df = pd.read_csv('C:/Users/Administrator/Favorites/Jupyter ML/train.csv') #read data


# In[3]:


df.describe()


# In[4]:


df.isnull().sum().sort_values(ascending = False) #check null values


# # Part 2 Choosing target and features

# ## In this section, we are going to choose target variable and feature variables for machine learning models. In addition, we are also spliting the data into test set and training set for the validations later on.
# ## Since this is a machine-learning-technique-oriented model, we are going to select features using our business sense. For a in-depth selection of feature variables, please refer to a data-exploration-oriented project.

# In[5]:


#chose target variable
y = df.SalePrice 


# In[6]:


#choose feature variables
features = ['LotArea','OverallQual','YearBuilt', 'OverallCond']
X = df[features]


# In[7]:


X.head() # take a look on our feature data


# # Part 3 First ML model: Decision Tree Model

# ## In this section, we are going to build a decision tree machine learning model.

# In[8]:


from sklearn.tree import DecisionTreeRegressor


# In[9]:


#define model
d_tree_model = DecisionTreeRegressor(random_state = 1) #random_state ensure the same return in each run


# In[10]:


#fit model
d_tree_model.fit(X, y)


# In[11]:


#make predictions
print("Making predicitons for the following 5 houses:")
print(X.head(5))
print("The predictions are:")
print(d_tree_model.predict(X.head(5)))
print("The actual prices are:")
print(y.head(5))


# ## Part 3.1 Validation on the Decision Tree model

# After we have trained the model, we want to know how well our model perform on real-world dataset. In this section, we will seperate the original dataset into trainning and validation dataset just for convenience. We re-start fitting our model using the traning set and test the model on the validation dataset using mean absolute error. Mean absolute error is used to measure the difference level of two datasets.

# In[17]:


# validation of the model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# seperate the trainning and validation dataset
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#fit model with t
d_tree_model_train = DecisionTreeRegressor(random_state = 1)
d_tree_model_train.fit(train_X, train_y)

# test the model on validation dataset
tree_predictions = d_tree_model_train.predict(val_X)
print(mean_absolute_error(val_y, tree_predictions))


# # Part 4 Second Machine Learning Model: Random Forest Model

# ## We are going to build a Random forest model in this section. 

# In[14]:


#model building
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(X,y)


# In[15]:


#make predictions
print("Making predicitons for the following 5 houses:")
print(X.head(5))
print("The predictions are:")
print(forest_model.predict(X.head(5)))


# ## Part 4.1 Validation on Random Forest Model

# After we have trained the model, we want to know how well our random forest model perform on real-world dataset. In this section, we will seperate the original dataset into trainning and validation dataset just for convenience. We re-start fitting our model using the traning set and test the model on the validation dataset. You can develop the model further by changing the parameters of the model by yourself.

# In[19]:


# validation of the model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# seperate the trainning and validation dataset
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#fit model with t
forest_model_train = RandomForestRegressor(random_state = 1)
forest_model_train.fit(train_X, train_y)

# test the model on validation dataset
forest_predictions = forest_model_train.predict(val_X)
print(mean_absolute_error(val_y, forest_predictions))


# Here we see that the random forst model has a better mean absolute error (26805) than the decision tree model (33255)
