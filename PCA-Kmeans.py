#!/usr/bin/env python
# coding: utf-8

# #  Optimizing Feature Extraction with Principal Component Analysis
# 
# ### Analysis focuses on using Principal Component Analysis (PCA) to reduce the dimensionality of the Clinical data while highlighting the patient characteristics, such as behavioral and socioeconomic factors.

# In[341]:


from platform import python_version
print("\n python version for K-NN classification analysis is ",python_version())


# ### Import Libraries

# In[342]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# ## Load the data

# In[343]:


pcadata=pd.read_csv("path_to_csv")
pcadata.head()


# In[344]:


pcadata.info()


# ### The original clinical dataset contains 50 features and 10,000 rows. Here, we are selecting only the continuous numeric features for the linear dimensionality reduction method and eliminating categorical and discrete variables from the original dataset. 

# In[345]:


dmrdata=pcadata.select_dtypes(exclude="object")
dmrdata.info()


# In[346]:


dmrdata.shape


# In[347]:


#Removing descrete variables since PCA is linear dimensionality reduction algorithm
dmrdata=dmrdata.select_dtypes(exclude=['int64'])
dmrdata.info()


# ### Inspect skewness

# In[348]:


# Plot histograms using Seaborn
features = ['Lat', 'Lng', 'Income', 'VitD_levels', 'Initial_days', 'TotalCharge', 'Additional_charges']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(dmrdata[feature], bins=30, kde=True, stat="density",  alpha=0.6)
    plt.title(f'Histogram and KDE of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


# In[349]:


warnings.filterwarnings('ignore')
sns.pairplot(dmrdata, diag_kind='kde')
# Show the plot
plt.show()


# Observe the scatterplot shape (whether it forms a straight line, an upward or downward pattern, or a widely dispersed pattern). The appearance of the slope (positive or negative) can help determine whether a linear relationship exists. If two variables do not exhibit a strong linear relationship, it doesn't necessarily mean they aren't related in other ways. A lack of strong linear trends suggests that the relationship between the variables may not be straightforward.
# 
# PCA helps uncover hidden relationships between variables by transforming the data into a new set of uncorrelated components. Weak linear relationships suggest that variables may be connected in more complex ways that simple linear models can't capture. These components capture the variance in the data, revealing patterns that may not be obvious in the original feature space.

# ## Pearson correlation matrix

# In[350]:


# Calculate the Pearson correlation matrix
correlation_matrix = dmrdata.corr(method='pearson')

# Display the correlation matrix
print(correlation_matrix)


# ### Data preprocessing before PCA

# By its very nature, PCA is sensitive to the presence of outliers and therefore also to the presence of gross errors in the datasets.
# 

# ### Outliers

# In[351]:


#Outliers in features
# Box plot for specified features
sns.boxplot(data=dmrdata[['Lat', 'Lng', 'Income', 'VitD_levels', 'Initial_days', 'TotalCharge', 'Additional_charges']])

# Show the plot
plt.title('Box Plot of Features')
#plt.xticks(rotation=45) 
plt.show()
plt.savefig('plot.pdf', bbox_inches='tight')


# ### Cap outliers

# In[352]:


# Create a copy of the original DataFrame
clean_data = dmrdata.copy()

# Define the features for capping
features = ['Lat', 'Lng', 'Income', 'VitD_levels', 'Initial_days', 'TotalCharge', 'Additional_charges']

# Cap outliers in the copied DataFrame
for feature in features:
    lower_bound = clean_data[feature].quantile(0.05)  # 5 percentile
    upper_bound = clean_data[feature].quantile(0.95)    # 95 percentile
    clean_data[feature] = clean_data[feature].clip(lower=lower_bound, upper=upper_bound)


# In[353]:


# Show the plot
sns.boxplot(data=clean_data[['Lat', 'Lng', 'Income', 'VitD_levels', 'Initial_days', 'TotalCharge', 'Additional_charges']])
plt.title('Box Plot of Features')
#plt.xticks(rotation=45) 
plt.show()


# In[354]:


clean_data.describe()


# In[355]:


#To check how skewed the features are
skewness = clean_data.skew()
print(skewness)


# ## Treat skewness by using square root transformation

# In[356]:


#Stabilize using square root transformation 
clean_data['Income_sqrt'] = np.sqrt(clean_data['Income'])
clean_data['Additional_charges_sqrt'] = np.sqrt(clean_data['Additional_charges'])


# In[357]:


clean_data.info()


# ### Clean data

# In[358]:


pcclean_data=clean_data[['Income_sqrt','VitD_levels','Initial_days','TotalCharge','Additional_charges_sqrt','Lat','Lng']]
pcclean_data.info()


# ### Summary statistics

# In[359]:


pcclean_data.describe()


# ### Normalization to ensure that each feature contributes equally to the analysis

# In[360]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_dmr=pd.DataFrame(scaler.fit_transform(pcclean_data),columns=pcclean_data.columns)


# ### Summary of scaled data

# In[361]:


scaled_dmr.describe().round()


# ##  Normality check-visualization

# In[362]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define the features for plotting
features = ['Lat', 'Lng', 'Income_sqrt', 'VitD_levels', 'Initial_days', 'TotalCharge', 'Additional_charges_sqrt']

# Plot histograms with KDE overlay
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)  # Adjust the grid size based on number of features
    sns.histplot(scaled_dmr[feature], bins=30, kde=True,  edgecolor='black')
    plt.title(f' {feature} ')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[363]:


#Export cleaned data to CSV
scaled_dmr.to_csv("path_to_file",index=True)


# In[364]:


scaled_dmr.shape


# ### Apply PCA
# #### The pca.components_ attribute of the PCA object contains the principal component vectors. Each row corresponds to a data point, and each column represents a principal component.

# In[365]:


#pip install pca


# In[366]:


from sklearn.decomposition import PCA
pca=PCA()
dmr_pc=pca.fit_transform(scaled_dmr)


# In[367]:


# loading matrix
loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                        index=scaled_dmr.columns)

# Display the loadings
print("Loadings (feature contributions to each principal component):")
print(loadings)


# ## eigenvalues (variance explained by each component)

# In[368]:


# variance of each components-from eigh class of numpy.linalg module 
from numpy.linalg import eigh

# Determine covariance matrix
cov_matrix = np.cov(dmr_pc, rowvar=False)
#print(cov_matrix)

# Determine eigenvalues and eigenvectors
egnvalues, egnvectors = eigh(cov_matrix)
print("eigen values are \n",egnvalues)

# Determine explained variance
#total_egnvalues = sum(egnvalues)
#var_exp_manual = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]
#var_exp_manual


# In[369]:


#Get exact eigenvalues from PCA class (sorted and aligned to PCA components)
eigenvalues = pca.explained_variance_

# Display the eigenvalues
print("eigen values:\n", eigenvalues)


# ### Proportion of total variance

# In[370]:


#Proportion of total variance
exp_var=pca.explained_variance_ratio_
print("Explained ratio for the components are \n ",exp_var)


# ## Plot the Cumulative explained variance

# In[371]:


#Plot the Cumulative explained variance
pcomponents=np.arange(pca.n_components_)+1
# Plot the explained variance
plt.figure(figsize=(8, 6))
plt.bar(pcomponents, exp_var, alpha=0.7, align='center', label='Individual explained variance')
plt.step(pcomponents, np.cumsum(exp_var), where='mid', label='Cumulative explained variance', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ### Scree plot

# In[372]:


# Scree plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', color='b')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.grid(True)
plt.show()


# ### The Kaiser rule is to drop all components with eigenvalues under 1.0 – this being the eigenvalue equal to the information accounted for by an average single item

# In[373]:


# Kaiser Criterion: components with eigenvalue > 1
kaiser_criterion = eigenvalues > 1
kaiser_components = np.sum(kaiser_criterion)

print(f"Number of components by Kaiser criterion: {kaiser_components}")

# Plot the eigenvalues to visualize
plt.figure(figsize=(8, 6))
plt.bar(np.arange(1, len(eigenvalues) + 1), eigenvalues)
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue = 1)')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Principal Components (Kaiser Criterion)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[374]:


# Find components with eigenvalue > 1 (Kaiser Criterion)
#The Kaiser rule is to drop all components with eigenvalues under 1.0
kaiser_criterion = np.where(eigenvalues > 1)[0]
print(f"Components to retain (Kaiser criterion): {kaiser_criterion + 1}")


# ##  Plot the loading matrix

# In[375]:


# Plot the loading matrix
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, cmap='coolwarm', center=0)
plt.title('Feature Contributions to Principal Components')
plt.show()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




