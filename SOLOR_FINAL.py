#!/usr/bin/env python
# coding: utf-8

# IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv('/home/aman/Documents/a.csv')
data.head(10)


# In[5]:


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(X.shape, y.shape)
y = np.reshape(y, (-1,1))
y.shape


# In[6]:


X


# In[7]:


y


# SPLITTING TEST AND TRANING DATA

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Training data Shape: {} {} \nTesting data Shape: {} {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))


# SCALING THE TRAINING DATA

# In[10]:


from sklearn.preprocessing import StandardScaler
# input scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# outcome scaling:
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)    
y_test = sc_y.transform(y_test)


# In[11]:


X_train


# In[12]:


X_test


# In[13]:


y_train


# In[14]:


y_test


# CREATING THE NEURAL NETWROK MODEL

# In[24]:


def create_model(n_layers, n_activation, kernels):
  model = tf.keras.models.Sequential()
  for i, nodes in enumerate(n_layers):
    if i==0:
      model.add(Dense(nodes, kernel_initializer=kernels, activation=n_activation, input_dim=X_train.shape[1]))
      #model.add(Dropout(0.2))
    else:
      model.add(Dense(nodes, activation=n_activation, kernel_initializer=kernels))
      #model.add(Dropout(0.4))
  
  model.add(Dense(1))
  model.compile(loss='mse', 
                optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model


# In[25]:


output_model = create_model([32,64],'relu','normal')
output_model.summary()


# In[26]:


from keras.utils.vis_utils import plot_model
plot_model(output_model, to_file='spfnet_model.png', show_shapes=True, show_layer_names=True)


# TRAINING THE MODEL

# In[27]:


hist = output_model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),epochs=200, verbose=2)


# PLOTTING TOOT MEAN SQAURE ERROR VS THE EPOCHS

# In[28]:


plt.plot(hist.history['root_mean_squared_error'])
#plt.plot(hist.history['val_root_mean_squared_error'])
plt.title('Root Mean Squares Error')
plt.xlabel('Epochs')
plt.ylabel('error')
plt.show()


# EVALUEATE THE MODEL
# 

# In[29]:


output_model.evaluate(X_train, y_train)


# MEAN SQUARED ERROR FOR THE X_train 

# In[31]:


from sklearn.metrics import mean_squared_error

y_pred = output_model.predict(X_test) # get model predictions (scaled inputs here)
y_pred_orig = sc_y.inverse_transform(y_pred) # unscale the predictions
y_test_orig = sc_y.inverse_transform(y_test) # unscale the true test outcomes

RMSE_orig = mean_squared_error(y_pred_orig, y_test_orig, squared=False)
RMSE_orig


# MEAN SQUARED ERROR FOR y_train

# In[33]:


train_pred = output_model.predict(X_train) # get model predictions (scaled inputs here)
train_pred_orig = sc_y.inverse_transform(train_pred) # unscale the predictions
y_train_orig = sc_y.inverse_transform(y_train) # unscale the true train outcomes

mean_squared_error(train_pred_orig, y_train_orig, squared=False)


# In[ ]:


r2 score for y_train


# In[34]:


from sklearn.metrics import r2_score
r2_score(y_pred_orig, y_test_orig)


# r2 score for X_train

# In[35]:


r2_score(train_pred_orig, y_train_orig)


# In[36]:


np.concatenate((train_pred_orig, y_train_orig), 1)


# In[37]:


np.concatenate((y_pred_orig, y_test_orig), 1)


# PLOTTING TEST DATA VS TRAINING DATA

# In[38]:


plt.figure(figsize=(16,6))
plt.subplot(1,2,2)
plt.scatter(y_pred_orig, y_test_orig)
plt.xlabel('Predicted Generated Power on Test Data')
plt.ylabel('Real Generated Power on Test Data')
plt.title('Test Predictions vs Real Data')
#plt.scatter(y_test_orig, sc_X.inverse_transform(X_test)[:,2], color='green')
plt.subplot(1,2,1)
plt.scatter(train_pred_orig, y_train_orig)
plt.xlabel('Predicted Generated Power on Training Data')
plt.ylabel('Real Generated Power on Training Data')
plt.title('Training Predictions vs Real Data')
plt.show()


#  SOLOR AZIMUTH TEST VS PREDICTED DATA

# In[39]:


x_axis = sc_X.inverse_transform(X_train)[:,-1]
x2_axis = sc_X.inverse_transform(X_test)[:,-1]
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.scatter(x_axis, y_train_orig, label='Real Generated Power')
plt.scatter(x_axis, train_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on Training Data')
plt.xlabel('Solar Azimuth')
plt.title('Training Predictions vs Solar Azimuth')
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.scatter(x2_axis, y_test_orig, label='Real Generated Power')
plt.scatter(x2_axis, y_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on TEST Data')
plt.xlabel('Solar Azimuth')
plt.title('TEST Predictions vs Solar Azimuth')
plt.legend(loc='lower right')
plt.show()


# mean_sea_level_pressure_MSL TEST VS PREDICTED DATA

# In[45]:


x_axis = sc_X.inverse_transform(X_train)[:,-17]
x2_axis = sc_X.inverse_transform(X_test)[:,-17]
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.scatter(x_axis, y_train_orig, label='Real Generated Power')
plt.scatter(x_axis, train_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on Training Data')
plt.xlabel('mean_sea_level_pressure_MSL')
plt.title('Training Predictions vs mean_sea_level_pressure_MSL')
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.scatter(x2_axis, y_test_orig, label='Real Generated Power')
plt.scatter(x2_axis, y_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on TEST Data')
plt.xlabel('mean_sea_level_pressure_MSL')
plt.title('TEST Predictions vs mean_sea_level_pressure_MSL')
plt.legend(loc='lower right')
plt.show()


# wind_speed_80_m_above_gnd TEST VS PREDICTED DATA

# In[58]:


x_axis = sc_X.inverse_transform(X_train)[:,-8]
x2_axis = sc_X.inverse_transform(X_test)[:,-8]
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.scatter(x_axis, y_train_orig, label='Real Generated Power')
plt.scatter(x_axis, train_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on Training Data')
plt.xlabel('wind_speed_80_m_above_gnd')
plt.title('Training Predictions vs wind_speed_80_m_above_gnd')
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.scatter(x2_axis, y_test_orig, label='Real Generated Power')
plt.scatter(x2_axis, y_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on TEST Data')
plt.xlabel('wind_speed_80_m_above_gnd')
plt.title('TEST Predictions vs wind_speed_80_m_above_gnd')
plt.legend(loc='lower right')
plt.show()


# In[47]:


sc = StandardScaler()
pred_whole = output_model.predict(sc.fit_transform(X))
pred_whole_orig = sc_y.inverse_transform(pred_whole)
pred_whole_orig


# In[48]:


y


# In[49]:


r2_score(pred_whole_orig, y)


# In[50]:


df_results = pd.DataFrame.from_dict({
    'R2 Score of Whole Data Frame': r2_score(pred_whole_orig, y),
    'R2 Score of Training Set': r2_score(train_pred_orig, y_train_orig),
    'R2 Score of Test Set': r2_score(y_pred_orig, y_test_orig),
    'Mean of Test Set': np.mean(y_pred_orig),
    'Standard Deviation pf Test Set': np.std(y_pred_orig),
    'Relative Standard Deviation': np.std(y_pred_orig) / np.mean(y_pred_orig),
},orient='index', columns=['Value'])
display(df_results.style.background_gradient(cmap='afmhot', axis=0))


# HEATMAP SHOWING FEATURE IMPORTANCE

# In[51]:


corr = dts.corr()
plt.figure(figsize=(22,22))
sns.heatmap(corr, annot=True, square=True);


# In[52]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.001)

lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

lasso_coeff = pd.DataFrame({'Feature Importance':lasso.coef_}, index=dts.columns[:-1])
lasso_coeff.sort_values('Feature Importance', ascending=False)


# In[53]:


g = lasso_coeff[lasso_coeff['Feature Importance']!=0].sort_values('Feature Importance').plot(kind='barh',figsize=(6,6), cmap='winter')


# In[ ]:




