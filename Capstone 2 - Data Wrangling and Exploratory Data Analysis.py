#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os


# In[4]:


import numpy as np


# # Importing data into our notebook

# In[5]:


df= pd.read_csv(f'file:///Users/dariastachowiak/Downloads/features_30_sec.csv%20-%20Sheet1.csv')


# In[6]:


print(df.head())


# In[7]:


print(df.head)


# # Converting to HTML for easier reading

# In[8]:


from IPython.core.display import HTML


# In[9]:


display(HTML(df.to_html()))


# # General insights about our data

# In[10]:


df.info()


# In[11]:


df.set_index('filename')


# In[12]:


df.isnull().sum().sum()


# Nice! No missing values in our dataframe

# # Beginning stages of data exploration

# Questions I will explore:
# 
#      Do I have an understanding of my data? Do I know the meaning of every column? 
# 
#     Is there any potentially erroneous data that could compromise the accuracy of the project further down the line?
#     
#     Do the general statistics for the genre categories make sense? As in, can the differences be rationally explained?
# 

# In[13]:


#Checking for outliers
outliers = df[df[df.columns.values] > df[df.columns.values].mean() + 3 * df[df.columns.values].std()]
display(HTML(outliers.to_html()))


# In[14]:


outliers_only=outliers.dropna(how = 'all') 


# In[15]:


display(HTML(outliers_only.to_html()))


# In[16]:


outliers_only1=outliers_only.dropna(axis=1,how='all')


# In[17]:


display(HTML(outliers_only1.to_html()))


# In[18]:


outliers_only1.shape


# In[19]:


outliers_count=pd.DataFrame(outliers_only1.count())
outliers_count1=outliers_count.reset_index(level=None, drop=False, inplace=False, col_level=0)
outliers_count1.sort_values(by=[0], ascending=False)


# In[20]:


outliers_count1['pct_total'] = (outliers_count1.iloc[:, 1] / 1000)*100


# In[21]:


print(outliers_count1.head())

# Since the data had been extracted from the audio via machine, we will be keeping the outliers in as it is unlikely that they are from a clerical error. Additionally, the percentage of outlier values for each feature is small(< or = to 3.1%).
# In[22]:


list(df.columns.values)


# # Defining features of audio:
# #chroma_stft_ : This is a chromagram from a waveform or power spectrogram. ransformation of a signal's time-frequency properties into a temporally varying precursor of pitch.
# #rms: This is the root-mean-square (RMS) value for each frame from each audio file. It is the the total value of the effective value
# #spectral_centroid: The spectral centroid is a measure used in digital signal processing to characterise a spectrum. It indicates where the center of mass of the spectrum is located
# #spectral_bandwidth: It is the Wavelength interval in which a radiated spectral quantity is not less than half its maximum value. It is a measure of the extent of the Spectrum.
# #rolloff: A frequency response which falls gradually above or below a certain frequency limit
# #zero_crossing_rate: The zero-crossing rate is the rate at which a signal changes from positive to zero to negative or from negative to zero to positive. 
# #harmony: Harmony arises in music when two or more notes sound at the same time. The values are the quantified dissimilarities between two sequences of musical chords.
# #tempo: The speed at which a passage of music is played
# #mfcc: In sound processing, the mel-frequency cepstrum is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. Mel-frequency cepstral coefficients are coefficients that collectively make up an MFC.

# # Diving deeper: genre statistics

# In[23]:


statistics=df.groupby('label').describe()


# In[24]:


print(statistics)


# In[25]:


display(HTML(statistics.to_html()))

Since there are only 10 genres to look at, I could "afford" to get an in-depth look at the statistics for each feature in order to, once again, make note of any potential "red flags" pertaining to the data. 

Immediately, I noticed that the Standard Deviation for the "length" feature was "0" for both blues and pop. For both of these genres, I looked back at the data to ensure that the "length" was indeed the same for all of the samples. After confirming this, I proceeded forward. 

# In[26]:


std_df=df.groupby(['label']).describe().loc[:,(slice(None),['std'])]


# In[27]:


display(HTML(std_df.to_html()))

Getting a closer look at the standard deviation allows us to look at variations within genres. For example, it makes sense that classical music has a much higher tempo variance than disco. It is widely known that classical music tends to encompass a variety of moods, speeds, and is incredibly diverse whereas disco has a pretty standard tempo (good for dancing) and therefore, the variety in tempo will be less significant. 

# # Exploratory Data Analysis 

# Questions I will explore in this section:
# 
#     Are there any features that are strongly correlated with each other? If so, what kind of relationship do they have? Answering this question could help us potentially predict a variable value when given a genre type.
#     
#     How do the genres compare for each of the features? I will create visualizations that will allow us to spot any major patterns. 

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn


# In[29]:


spike_cols = [col for col in df.columns]
corr = df[spike_cols].corr()

f, ax = plt.subplots(figsize=(27, 20));

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True


sns.heatmap(corr,mask=mask, cmap="coolwarm", vmax=1.0, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 1.0})

plt.title('Correlation Heatmap for Audio Variables', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);
plt.savefig("Heatmap.jpg")


# From this heatmap, we can find that many of our features have a medium to strong (>.75) correlation. It makes sense that all of the mel-frequency cepstral coefficients are strongly correlated with each other as they all make up the mel-frequency cepstrum (MFC), which a representation of the short-term power spectrum of a sound.
# 
# We also find that the rolloff mean, the zero crossing rate mean and the spectral bandwidth mean are strongly postively correlated with the spectral centroid mean. 
# 
# The Mfcc 2 mean is strongly negatively correlated with the spectral centroid mean, spectral bandwidth mean and the rolloff mean. This is interesting as this means that the second frame of each song (1.33 sec to 1.99 sec ) has unique characteristics when compared to other frames.

# In[110]:


df_sc=df.drop('filename',axis=1)


# In[111]:


df_sc.head()


# In[112]:


df_sc.set_index('label')


# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns

for column in df_sc.columns[1:]:
    sns.set()
    fig, ax = plt.subplots()
    sns.set(style="ticks")
    sns.violinplot(x='label', y=column, data=df_sc) 
    sns.despine(offset=10, trim=True) 
    fig.set_size_inches(10,5)
  


# A series of violin plots lets us visually compare the spread of each feature for every genre. 
# 
# Some observations that stood out to me include:
# 
# 1.) The rms variance (which, to describe simply, gives us the variance of 'sound power' through the song) varies drastically for hip hop and pop vs genres such as classical, country, rock, metal and jazz. In general, songs within this genre tend to have a more similar and more 'defined' sound vs. hip hop and pop. Many of today's songs fall into these 2 broad categories. 
# 
# 
# If we look at the rms mean, on the other hand, we find that the distributions are more similar, with pop continuing to be the most varied category and classical music. This makes sense as the rms is averaged out, bringing all the generes a bit closer together.
# 
# 2.) Metal and classical songs are most similar in their respective genres when it comes to rolloff variance. Hip hop, pop, reggae, and rock all have the widest distibutions.
# 
# 3.) Reggae has a significantly wider distribution than the rest of the genres when it comes to zero crossing rate variance. Zero crossing rate can be used to describe the "smoothness" of a sound. For example, speech has a higher variance and music has a lower variance. It is possible that because reggae songs include a distinctive style of singing that this is why we see such a large spread. Nevertheless, the mean for reggae is on the low end with metal rising to the top. This makes sense as metal is not known for its particularly "smooth" sound....
# 
# 4.) Jazz has the widest spread of spectral bandwidth mean and classical has the lowest mean with one of the tightest spreads(besides metal). Pop music has the highest spectral bandwidth mean. If a signal is made up of many high frequencies, the bandwidth will be large, and if the signal is made up of low frequencies, the bandwidth will be small. Overall, we can conclude that lower frequenceies are more prevelant in classical music whereas pop has the highest frequencies.
# 
# 5.)Overall, based the observations above, it is likely that pop will be one of the more difficult genres to classify and classical and metal will likely be the easiest. These two genres seem to be fairly distinct. 
# 
# 

# As we see, we have 59 variables in this dataframe so it would be useful to utilize a dimension reduction technique. In this case, PCA would be the most popular choice. Let's see if it would be able to explain the data sufficiently...

# First step is standardizing the data:

# In[99]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[100]:


features=['length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
       'spectral_centroid_mean', 'spectral_centroid_var',
       'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
       'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
       'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']
x = df_sc.loc[:, features].values
y = df_sc.loc[:,['label']].values

x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = features).head()


# We are going to reduce our data to 2 dimensions:

# In[101]:


pca = PCA(n_components=2)


# In[102]:


principalComponents = pca.fit_transform(x)


# In[103]:


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[104]:


principalDf.head(5)


# In[105]:


df[['label']].head()


# In[106]:


finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
finalDf.head(5)


# Next, we are going to plot the data to visualize it:

# In[107]:


import plotly.express as px
fig = px.scatter(principalComponents, x=0, y=1, color=df['label'])

fig.show()


# Based on the graph above, we can say that our 2 features give SOMEWHAT of an explanation for our genres, however, it doesn't seem like nearly enough to be able to explain our data. The graph is looking pretty muddy...

# In[108]:


pca.explained_variance_ratio_


# As predicted, the 2 features only explain ~41% of our data.

# In[109]:


pca = PCA()
pca.fit(pd.DataFrame(data = x, columns = features))


exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

px.area(
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul,
    labels={"x": "# Components", "y": "Explained Variance"})


# It seems like it would be most useful to include ~12 components in our data as this is when the explained variance finally crosses the 80% threshold. This information may be useful as we begin fitting models to best classify our data...

# In[ ]:




