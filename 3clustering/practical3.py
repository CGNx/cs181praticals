
# coding: utf-8

# ##Preparing Data

# In[1]:

from modules import *
import sklearn


# In[2]:

artists = pd.read_csv('data/artists.csv.gz', compression = 'gzip')
profiles = pd.read_csv('data/profiles.csv.gz', compression = 'gzip')
test = pd.read_csv('data/test.csv.gz', compression = 'gzip')
train = pd.read_csv('data/train.csv.gz', compression = 'gzip')


# In[3]:

data = pd.merge(left = train, right = profiles, how = 'left', on ='user')


# In[4]:

gb = data.groupby('user')
a = gb.get_group('894484ed3c451c4468b043f363a4caf6de30f9ef')
b = gb.get_group('528200ac6ac9a35857d8952a26cf6f4738960643')
pd.merge(left = a, right = b, how = 'inner', on = 'artist', )


# In[5]:

demog = data.copy(deep = True)


# In[6]:

demog = demog.drop('user', axis = 1)
demog['age'] = demog.age.replace(np.nan, 0) 
demog['sex'] = demog.sex.replace('NaN', -1) 
demog['sex'] = [1 if i == 'm' else (0 if i == 'f' else -1) for i in demog.sex]


# In[7]:

testdf = pd.merge(left = test, right = profiles, how = 'left', on ='user')


# In[8]:

testdf = testdf.drop('user', axis = 1)
testdf['age'] = testdf.age.replace(np.nan, 0) 
testdf['sex'] = testdf.sex.replace('NaN', -1) 
testdf['sex'] = [1 if i == 'm' else (0 if i == 'f' else -1) for i in testdf.sex]
testdf.head()


# ##min/diff
# 
# ##General algorithm
# 1. For sex, age, country of user, find k-nearest neighbors that that have that country and that artist.
# 2. average plays of k-nearest neighbors.
# 3. weight with prior of average plays for that artist
# 4. return result

# In[9]:

#clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
clf = sklearn.neighbors.NearestNeighbors()


# In[10]:

prior = demog.groupby('artist').plays.mean()


# In[ ]:

gb = demog.groupby(['artist','country'])


# In[ ]:

t = datetime.now()
plays = []
for i, row in testdf.iterrows():
    result = 0
    try:
        index = tuple(row[['artist', 'country']])
        df = gb.get_group(index)
        df = df.reset_index()
        dist, indices = clf.fit(df[['sex','age']]).kneighbors(row[['sex', 'age']],n_neighbors = min(5, len(df)))
        result = df.ix[indices[0]].plays.mean() #Mean of k-nearest neighbors plays
        result = .2 * prior.ix[row['artist']] + .8*result #Incorporate prior belief
    except:
        #print (i, row)
        result = prior.ix[row['artist']]
    plays.append([i, result])
    if np.random.randint(10000) == 0:
        elapsed = datetime.now() - t
        total = (elapsed / i) * len(testdf)
        print "Time Remaining:", total - elapsed


# In[14]:

result = pd.DataFrame(plays)
result = result.apply(pd.Series.round)
result.columns = ['Id', 'Plays']
result.Id = result.Id + 1
result.to_csv('submission1.csv', index = False)


# In[34]:

#Trying removing prior
bias = []
for i, row in testdf.iterrows():
    bias.append(.2 * prior.ix[row['artist']])


# In[37]:

bias = pd.Series(bias)
result.Plays = (result.Plays - bias)*1.25


# In[40]:

result = result.apply(pd.Series.round)
result.to_csv('submission2.csv', index = False)


# In[41]:

prior.head()


# In[42]:

testdf.head()


# In[44]:

sub3 = testdf.artist.replace(prior)


# In[46]:

sub3 = pd.DataFrame(sub3).reset_index()


# In[48]:

sub3.columns = ['Id','Plays']
sub3['Id'] = sub3.Id + 1


# In[49]:

sub3 = sub3.apply(pd.Series.round)


# In[52]:

sub3.to_csv('submission3.csv', index = False)


# In[ ]:



