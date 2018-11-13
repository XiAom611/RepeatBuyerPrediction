
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize


# In[2]:


userLogInput = "../data/user_log_format1.csv"
userInfoInput = "../data/user_info_format1.csv"


# In[3]:


userLog = pd.read_csv(userLogInput, low_memory=False)
userInfo = pd.read_csv(userInfoInput, low_memory=False)


# In[4]:


userInfo.age_range.fillna(0,inplace=True)
userInfo.age_range.replace(8,7,inplace=True)


# In[5]:


userInfo.gender.fillna(2, inplace=True)


# In[6]:


userLog.head()


# In[7]:


userInfo.head()


# In[8]:


u_age = userInfo[['user_id','age_range']].copy()
u_age = u_age.drop_duplicates()
u_age.age_range = u_age.age_range.astype(int)


age_hot = label_binarize(np.array(u_age.age_range), classes=[0,1,2,3,4,5,6,7])

u_age['age_range_0'] = age_hot[:,0]
u_age['age_range_1'] = age_hot[:,1]
u_age['age_range_2'] = age_hot[:,2]
u_age['age_range_3'] = age_hot[:,3]
u_age['age_range_4'] = age_hot[:,4]
u_age['age_range_5'] = age_hot[:,5]
u_age['age_range_6'] = age_hot[:,6]
u_age['age_range_7'] = age_hot[:,7]
u_age.drop(columns = ['age_range'], inplace=True)

print(u_age.isnull().any())
print(u_age.shape)
u_age.head()


# In[9]:


u_gender = userInfo[['user_id','gender']].copy()
u_gender = u_gender.drop_duplicates(['user_id'])

gender_hot = label_binarize(np.array(u_gender.gender), classes=[0, 1, 2])
u_gender['gender_hot_0'] = gender_hot[:,0]
u_gender['gender_hot_1'] = gender_hot[:,1]
u_gender['gender_hot_2'] = gender_hot[:,2]
u_gender.drop(columns=['gender'],inplace=True)

print(u_gender.isnull().any())
print(u_gender.shape)
u_gender.head()


# In[10]:


u_actions = userLog[['user_id','action_type']].copy()
u_actions.action_type = 1
u_actions = u_actions.groupby(['user_id']).agg('sum').reset_index()
u_actions = u_actions.rename(index=str,columns={'action_type':'u_total_actions'})
print('total_actions',u_actions.shape)
u_actions.head()


# In[11]:


u_actionsType = userLog[['user_id','action_type']].copy()
action_type_hot = label_binarize(np.array(u_actionsType.action_type), classes=[0, 1, 2, 3])

u_actionsType['u_action_type_hot_0'] = action_type_hot[:,0]
u_actionsType['u_action_type_hot_1'] = action_type_hot[:,1]
u_actionsType['u_action_type_hot_2'] = action_type_hot[:,2]
u_actionsType['u_action_type_hot_3'] = action_type_hot[:,3]

u_actionsType.drop(columns=['action_type'],inplace=True)

u_actionsType = u_actionsType.groupby(['user_id']).agg('sum').reset_index()
print(u_actionsType.shape)
u_actionsType.head()


# In[12]:


u_days = userLog[['user_id','time_stamp']].copy()
u_days.drop_duplicates(inplace=True)

u_days.time_stamp = 1
u_days = u_days.groupby(['user_id']).agg('sum').reset_index()
u_days = u_days.rename(index=str,columns={'time_stamp':'u_days'})
print('u_days',u_days.shape)
u_days.head()


# In[13]:


u_items = userLog[['user_id','item_id']].copy()
u_items.drop_duplicates(inplace=True)

u_items.item_id = 1
u_items = u_items.groupby(['user_id']).agg('sum').reset_index()
u_items = u_items.rename(index=str,columns={'item_id':'u_items'})
print('u_items',u_items.shape)
u_items.head()


# In[14]:


u_cats = userLog[['user_id','cat_id']].copy()
u_cats.drop_duplicates(inplace=True)

u_cats.cat_id = 1
u_cats = u_cats.groupby(['user_id']).agg('sum').reset_index()
u_cats = u_cats.rename(index=str,columns={'cat_id':'u_cats'})
print(u_cats.shape)
u_cats.head()


# In[15]:


u12 = pd.merge(u_age, u_gender, on=['user_id'])

u123 = pd.merge(u12, u_actionsType, on=['user_id'])

u1234 = pd.merge(u123, u_actions, on=['user_id'])

u12345 = pd.merge(u1234, u_days, on=['user_id'])

u123456 = pd.merge(u12345, u_items, on=['user_id'])

u1234567 = pd.merge(u123456, u_cats, on=['user_id'])

userFeature = u1234567

print(userFeature.isnull().any())
print(userFeature.shape)
userFeature.head()


# In[16]:


s_actions = userLog[['seller_id','action_type']].copy()
s_actions.action_type = 1
s_actions = s_actions.groupby(['seller_id']).agg('sum').reset_index()
s_actions = s_actions.rename(index=str,columns={'action_type':'s_total_actions'})
print('total_actions',s_actions.shape)
s_actions.head()


# In[17]:


s_actionsType = userLog[['seller_id','action_type']].copy()
action_type_hot = label_binarize(np.array(s_actionsType.action_type), classes=[0, 1, 2, 3])

s_actionsType['s_action_type_hot_0'] = action_type_hot[:,0]
s_actionsType['s_action_type_hot_1'] = action_type_hot[:,1]
s_actionsType['s_action_type_hot_2'] = action_type_hot[:,2]
s_actionsType['s_action_type_hot_3'] = action_type_hot[:,3]

s_actionsType.drop(columns=['action_type'],inplace=True)

s_actionsType = s_actionsType.groupby(['seller_id']).agg('sum').reset_index()
print(u_actionsType.shape)
s_actionsType.head()


# In[18]:


s_items = userLog[['seller_id','item_id']].copy()
s_items['items_of_the_seller'] = 1

s_items.drop_duplicates(['seller_id','item_id'],inplace=True)
s_items = s_items.groupby(['seller_id']).agg('sum').reset_index()
s_items.drop(columns=['item_id'],inplace=True)

print(s_items.shape)
s_items.head()


# In[19]:


s_cats = userLog[['seller_id','cat_id']].copy()
s_cats['cats_of_the_seller'] = 1

s_cats.drop_duplicates(['seller_id','cat_id'],inplace=True)
s_cats = s_cats.groupby(['seller_id']).agg('sum').reset_index()
s_cats.drop(columns=['cat_id'],inplace=True)

print(s_cats.shape)
s_cats.head()


# In[20]:


s12 = pd.merge(s_actions, s_actionsType, on=['seller_id'])

s123 = pd.merge(s12, s_items, on=['seller_id'])

s1234 = pd.merge(s123, s_cats, on=['seller_id'])



# In[21]:


sellerFeature = s1234
print(sellerFeature.isnull().any())
print(sellerFeature.shape)
sellerFeature.head()


# In[22]:


us_actions = userLog[['user_id','seller_id','action_type']].copy()
us_actions.action_type = 1
us_actions = us_actions.groupby(['user_id','seller_id']).agg('sum').reset_index()
us_actions = us_actions.rename(index=str,columns={'action_type':'us_actions'})
print(us_actions.shape)
us_actions.head()


# In[23]:


us_actions_type = userLog[['user_id','seller_id','action_type']].copy()

action_type_hot = label_binarize(np.array(us_actions_type.action_type), classes=[0, 1, 2, 3])

us_actions_type['us_action_type_hot_0'] = action_type_hot[:,0]
us_actions_type['us_action_type_hot_1'] = action_type_hot[:,1]
us_actions_type['us_action_type_hot_2'] = action_type_hot[:,2]
us_actions_type['us_action_type_hot_3'] = action_type_hot[:,3]

us_actions_type.drop(columns=['action_type'],inplace=True)


us_actions_type = us_actions_type.groupby(['user_id','seller_id']).agg('sum').reset_index()


print(us_actions_type.shape)
us_actions_type.head()


# In[24]:


us_days = userLog[['user_id','seller_id','time_stamp']].copy()
us_days.drop_duplicates(inplace=True)

us_days.time_stamp = 1
us_days = us_days.groupby(['user_id','seller_id']).agg('sum').reset_index()
us_days = us_days.rename(index=str,columns={'time_stamp':'us_days'})

print(us_days.shape)
us_days.head()


# In[25]:


us_items = userLog[['user_id','seller_id','item_id']].copy()
us_items.drop_duplicates(inplace=True)

us_items.item_id = 1
us_items = us_items.groupby(['user_id','seller_id']).agg('sum').reset_index()
us_items = us_items.rename(index=str,columns={'item_id':'us_items'})

print(us_items.shape)
us_items.head()


# In[26]:


us_cats = userLog[['user_id','seller_id','cat_id']].copy()
us_cats.drop_duplicates(inplace = True)

us_cats.cat_id = 1
us_cats = us_cats.groupby(['user_id','seller_id']).agg('sum').reset_index()
us_cats = us_cats.rename(index=str,columns={'cat_id':'us_cats'})

print(us_cats.shape)
us_cats.head()


# In[27]:


us1 = pd.merge(us_actions, us_actions_type, on=['user_id', 'seller_id'], how='left')

us12 = pd.merge(us1, us_items, on=['user_id','seller_id'], how='left')

us123 = pd.merge(us12, us_days, on = ['user_id', 'seller_id'], how='left')

us1234 = pd.merge(us123, us_cats, on = ['user_id', 'seller_id'], how='left')




# In[28]:


usFeature = us1234


# In[29]:


print(usFeature.isnull().any())
print(usFeature.shape)
usFeature.head()


# In[30]:


userFeature.to_csv("../data/userFeatures.csv")


# In[31]:


sellerFeature.to_csv("../data/sellerFeatures.csv")


# In[32]:


usFeature.to_csv("../data/usFeatures.csv")


# In[26]:


userFeature=pd.read_csv("../data/userFeatures.csv")
sellerFeature=pd.read_csv("../data/sellerFeatures.csv")
usFeature=pd.read_csv("../data/usFeatures.csv")


# In[27]:


allTrainData = usFeature[['user_id','seller_id']].copy()
allTrainData.drop_duplicates(inplace=True)
print(allTrainData.shape)

allTrainData = pd.merge(allTrainData, userFeature, on=['user_id'])
print(allTrainData.shape)


allTrainData = pd.merge(allTrainData, sellerFeature, on=['seller_id'])
print(allTrainData.shape)


allTrainData = pd.merge(allTrainData, usFeature, on=['user_id','seller_id'])
print(allTrainData.shape)


# In[28]:


from sklearn import preprocessing


# In[29]:


allTrainData = allTrainData.sort_values(by=['user_id','seller_id']).reset_index()


# In[30]:


allTrainData.head()


# In[31]:


allTrainData.isnull().any()


# In[32]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# In[33]:


trainPath = "../data/train_format1.csv"
testPath = "../data/test_format1.csv"


# In[34]:


trainData = pd.read_csv(trainPath, low_memory=False)
testData = pd.read_csv(testPath, low_memory=False)


# In[35]:


trainData.shape


# In[36]:


trainData.head()


# In[37]:


testData.shape


# In[38]:


testData.head()


# In[39]:


trainData = trainData.rename(index=str,columns={'merchant_id':'seller_id'})
testData = testData.rename(index=str,columns={'merchant_id':'seller_id'})


# In[40]:


dataset = allTrainData.iloc[:,1:]


# In[41]:


t_dataset = pd.merge(dataset,trainData,on=['user_id','seller_id'],how='right')
t_dataset.drop(columns=['user_id','seller_id'],inplace=True)
print('t_dataset',t_dataset.shape)
t_dataset.isnull().any()


# In[42]:


c_dataset = pd.merge(dataset,testData,on=['user_id','seller_id'],how='right')

c_dataset.drop(columns=['user_id','seller_id'],inplace=True)
X_challenge = c_dataset.iloc[:,0:-1]

X_challenge.count()


# In[48]:


X_challenge=X_challenge.drop(columns=['Unnamed: 0_x','Unnamed: 0_y','Unnamed: 0'])


# In[49]:


X_challenge.shape


# In[50]:


X_challenge.head()


# In[55]:


X_challenge_scaled = preprocessing.scale(X_challenge)
X_challenge.age_range_0 = X_challenge_scaled[:,0]       
X_challenge.age_range_1 = X_challenge_scaled[:,1]
X_challenge.age_range_2 = X_challenge_scaled[:,2]
X_challenge.age_range_3 = X_challenge_scaled[:,3]
X_challenge.age_range_4 = X_challenge_scaled[:,4]
X_challenge.age_range_5 = X_challenge_scaled[:,5]
X_challenge.age_range_6 = X_challenge_scaled[:,6]
X_challenge.age_range_7 = X_challenge_scaled[:,7]
X_challenge.gender_hot_0= X_challenge_scaled[:,8]
X_challenge.gender_hot_1= X_challenge_scaled[:,9]
X_challenge.gender_hot_2= X_challenge_scaled[:,10]
X_challenge.u_action_type_hot_0= X_challenge_scaled[:,11]
X_challenge.u_action_type_hot_1= X_challenge_scaled[:,12]
X_challenge.u_action_type_hot_2= X_challenge_scaled[:,13]
X_challenge.u_action_type_hot_3= X_challenge_scaled[:,14]
X_challenge.u_total_actions= X_challenge_scaled[:,15]
X_challenge.u_days  = X_challenge_scaled[:,16]
X_challenge.u_items = X_challenge_scaled[:,17]
X_challenge.u_cats  = X_challenge_scaled[:,18]
X_challenge.s_total_actions= X_challenge_scaled[:,19]
X_challenge.s_action_type_hot_0= X_challenge_scaled[:,20]
X_challenge.s_action_type_hot_1= X_challenge_scaled[:,21]
X_challenge.s_action_type_hot_2= X_challenge_scaled[:,22]
X_challenge.s_action_type_hot_3 = X_challenge_scaled[:,23]
X_challenge.items_of_the_seller = X_challenge_scaled[:,24]
X_challenge.cats_of_the_seller  = X_challenge_scaled[:,25]
X_challenge.us_actions          = X_challenge_scaled[:,26]
X_challenge.us_action_type_hot_0= X_challenge_scaled[:,27]
X_challenge.us_action_type_hot_1= X_challenge_scaled[:,28]
X_challenge.us_action_type_hot_2= X_challenge_scaled[:,29]
X_challenge.us_action_type_hot_3= X_challenge_scaled[:,30]
X_challenge.us_items            = X_challenge_scaled[:,31]
X_challenge.us_days             = X_challenge_scaled[:,32]
X_challenge.us_cats             = X_challenge_scaled[:,33]


# In[56]:


X_challenge_scaled.shape


# In[57]:


X_challenge_scaled


# In[58]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[59]:


X = t_dataset.iloc[:,0:-1]
Y = t_dataset.iloc[:,-1]


# In[66]:


X=X.drop(columns=['Unnamed: 0_x','Unnamed: 0_y','Unnamed: 0'])


# In[67]:


X_scaled = preprocessing.scale(X)
X.age_range_0 = X_scaled[:,0]       
X.age_range_1 = X_scaled[:,1]
X.age_range_2 = X_scaled[:,2]
X.age_range_3 = X_scaled[:,3]
X.age_range_4 = X_scaled[:,4]
X.age_range_5 = X_scaled[:,5]
X.age_range_6 = X_scaled[:,6]
X.age_range_7 = X_scaled[:,7]
X.gender_hot_0= X_scaled[:,8]
X.gender_hot_1= X_scaled[:,9]
X.gender_hot_2= X_scaled[:,10]
X.u_action_type_hot_0= X_scaled[:,11]
X.u_action_type_hot_1= X_scaled[:,12]
X.u_action_type_hot_2= X_scaled[:,13]
X.u_action_type_hot_3= X_scaled[:,14]
X.u_total_actions= X_scaled[:,15]
X.u_days  = X_scaled[:,16]
X.u_items = X_scaled[:,17]
X.u_cats  = X_scaled[:,18]
X.s_total_actions= X_scaled[:,19]
X.s_action_type_hot_0= X_scaled[:,20]
X.s_action_type_hot_1= X_scaled[:,21]
X.s_action_type_hot_2= X_scaled[:,22]
X.s_action_type_hot_3 = X_scaled[:,23]
X.items_of_the_seller = X_scaled[:,24]
X.cats_of_the_seller  = X_scaled[:,25]
X.us_actions          = X_scaled[:,26]
X.us_action_type_hot_0= X_scaled[:,27]
X.us_action_type_hot_1= X_scaled[:,28]
X.us_action_type_hot_2= X_scaled[:,29]
X.us_action_type_hot_3= X_scaled[:,30]
X.us_items            = X_scaled[:,31]
X.us_days             = X_scaled[:,32]
X.us_cats             = X_scaled[:,33]


# In[68]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=100)


# In[69]:


X_test.head()


# In[70]:


cls = RandomForestClassifier()


# In[71]:


cls.fit(X_train, Y_train)
print(roc_auc_score(Y_test,cls.predict_proba(X_test)[:,1]))
cls.predict_proba(X_test)


# In[72]:


pred = cls.predict_proba(X_challenge)


# In[73]:


pred


# In[74]:


output = pd.read_csv("../data/test_format1.csv")
output.head()


# In[75]:


output.prob = pred[:,1]


# In[76]:


output.head()


# In[77]:


output.to_csv("../data/RFCoutput0.csv",index=False)


# In[111]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(knn.predict_proba(X_test))

print(roc_auc_score(Y_test,knn.predict_proba(X_test)[:,1]))



# In[ ]:


knn_result = knn.predict_proba(X_challenge)


# In[114]:


output = pd.read_csv("../data/test_format1.csv")
output.head()
output.prob = knn_result[:,1]
output.to_csv("../data/knn_result.csv",index=False)


# In[159]:



from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[160]:


xgb = XGBClassifier()
xgb.fit(X_train, Y_train)


# In[161]:


print(roc_auc_score(Y_test,xgb.predict_proba(X_test)[:,1]))
xgb.predict_proba(X_test)


# In[162]:


xgb_pred = xgb.predict_proba(X_challenge)


# In[163]:


output = pd.read_csv("../data/test_format1.csv")
output.prob = xgb_pred[:,1]
output.to_csv("../data/XGBresult.csv",index=False)


# In[120]:


gbm = GradientBoostingClassifier(max_features='sqrt',
                                  learning_rate=0.1,
                                  min_samples_leaf=20,
                                  subsample=0.8,
                                  random_state=10,
                                  n_estimators=100)
gbm.fit(X_train,Y_train)
y_predprob = gbm.predict_proba(X_test)
print(roc_auc_score(Y_test,y_predprob[:,1]))


# In[121]:


gbm_result = gbm.predict_proba(X_challenge)


# In[123]:


output = pd.read_csv("../data/test_format1.csv")
output.prob = gbm_result[:,1]
output.to_csv("../data/GBMresult.csv",index=False)


# In[131]:


from sklearn.linear_model import LogisticRegression   


# In[132]:


lr = LogisticRegression()


# In[133]:


lr.fit(X_train, Y_train)
y_predprob = lr.predict_proba(X_test)
print(roc_auc_score(Y_test,y_predprob[:,1]))


# In[134]:


lr_result = lr.predict_proba(X_challenge)


# In[135]:


output = pd.read_csv("../data/test_format1.csv")
output.prob = gbm_result[:,1]
output.to_csv("../data/LRresult.csv",index=False)

