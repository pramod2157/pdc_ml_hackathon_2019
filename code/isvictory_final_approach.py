import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
le = preprocessing.LabelEncoder()

data_dir = '/home/pramod/pramod_work/pdc_hackathon/data/pdc_ml_hackathon_2019-master/data/'
train_data = pd.read_json(data_dir+'train.json', encoding='ascii')
train_data.to_csv(data_dir+'train.csv', index=False)
validation_data = pd.read_json(data_dir+'validation.json', encoding='ascii')
validation_data.to_csv(data_dir+'validation.csv', index=False)
train_data = pd.read_csv(data_dir+"train.csv")
train_y1 = train_data.petition_category
train_y2 = train_data.petition_is_victory
del train_data['petition_category']
del train_data['petition_is_victory']
clf = RandomForestClassifier(max_depth=2, random_state=21)
train_y1 = le.fit_transform(train_y1)
train_y2 = le.fit_transform(train_y2)
#initially tring for some of features following taking each feature at a time
train_data[["_score","petition_calculated_goal","petition_displayed_signature_count","petition_progress","petition_total_signature_count","petition_weekly_signature_count","petition_primary_target_is_person","petition_primary_target_publicly_visible","petition_primary_target_type","_source_coachable","_source_discoverable","_source_sponsored_campaign"]]
clf = LogisticRegression(random_state=  21)
columns = []
accuracy = []
train_y = train_y2
c=0
for col in train_x.columns:
    print(col)
    columns.append(col)
    train_x1 = train_x[col]
    if(c<5):
        if((any(train_x1.isnull()))|(any(train_x1.isna()))):
            train_x1[np.where(train_x1=='None')[0]] = (pd.DataFrame(train_x1)).median().iloc[0,0]
    if(c>5):
        if(train_x1.dtype!='bool'):
            if(any(train_x1=='None')):
                train_x1[np.where(train_x1=='None')[0]] = (pd.DataFrame(train_x1)).mode().iloc[0,0]
    X_train, X_test, y_train, y_test = train_test_split(train_x1, train_y, test_size=0.33, random_state=42)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    model= clf.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(f1_score(y_test, y_pred))
    accuracy.append(f1_score(y_test,y_pred))
    c=c+1

########undersampling majority class examples to deal with data imbalance########
indices = np.concatenate([random.sample(np.where(train_y2==0)[0],len(np.where(train_y2==1)[0])),np.where(train_y2==1)[0]])
pd.DataFrame(indices).to_csv(data_dir+"indices_usedafterUnderSampling.csv")
train_data_undersampled = train_data.iloc[indices,:]
#,"petition_primary_target_type","petition_user_country_code"
train_y2_undersampled = train_y2[indices]
train_x = train_data_undersampled[["_score","petition_calculated_goal","petition_displayed_signature_count","petition_progress","petition_total_signature_count","petition_weekly_signature_count","petition_primary_target_is_person","petition_primary_target_publicly_visible","_source_coachable","petition_primary_target_type",]]
#clf = LogisticRegression(random_state=  21)
columns = []
accuracy = []
train_y = train_y2_undersampled
c=0
train_data_undersampled_preprocessed = pd.DataFrame()
for col in train_x.columns:
    print(col)
    columns.append(col)
    train_x1 = train_x[col]
    if(c<5):
        if((any(train_x1.isnull()))|(any(train_x1.isna()))):
            train_x1[np.where(train_x1=='None')[0]] = (pd.DataFrame(train_x1)).median().iloc[0,0]
    if(c>5):
        if(train_x1.dtype!='bool'):
            if(any(train_x1=='None')):
                train_x1[np.where(train_x1=='None')[0]] = (pd.DataFrame(train_x1)).mode().iloc[0,0]
    if(col=="petition_primary_target_type"):
        train_x1 = le.fit_transform(train_x1)
    X_train, X_test, y_train, y_test = train_test_split(train_x1, train_y, test_size=0.33, random_state=42)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    model= clf.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(f1_score(y_test, y_pred))
    accuracy.append(f1_score(y_test,y_pred))
    c=c+1
    train_data_undersampled_preprocessed = pd.concat([train_data_undersampled_preprocessed,train_x1],axis=1)
train_data_undersampled_preprocessed = pd.concat([train_data_undersampled_preprocessed,train_y2_undersampled],axis=1)
train_data_undersampled_preprocessed.to_csv(data_dir +"train_data_undersampled_preprocessed_8.csv")
pd.DataFrame(train_x1).to_csv(data_dir+"undersampled_preprocessed_9thFeature.csv")

#overall results
clf = RandomForestClassifier(max_depth=2, random_state=21)
train_data_undersampled_preprocessed = pd.read_csv(data_dir+"train_data_undersampled_preprocessed_8.csv")
train_y = train_data_undersampled_preprocessed.petition_is_victory
del train_data_undersampled_preprocessed['petition_is_victory']
train_x = train_data_undersampled_preprocessed
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
#X_train = X_train.reshape(-1,1)
#X_test = X_test.reshape(-1,1)
model= clf.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(f1_score(y_test, y_pred))
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
c=0
test_data = pd.read_csv(data_dir+"validation.csv")
test_data = test_data[["_score","petition_calculated_goal","petition_displayed_signature_count","petition_progress","petition_total_signature_count","petition_weekly_signature_count","petition_primary_target_publicly_visible","_source_coachable","petition_primary_target_type",]]
test_data_prepro=pd.DataFrame()
for col in test_data.columns:
    print(col)
    test_x1 = test_data[col]
    if(c<5):
        if((any(test_x1.isnull()))|(any(test_x1.isna()))):
            train_x1[np.where(test_x1=='None')[0]] = (pd.DataFrame(test_x1)).median().iloc[0,0]
    if(c>5):
        if(test_x1.dtype!='bool'):
            if(any(test_x1=='None')):
                test_x1[np.where(test_x1=='None')[0]] = (pd.DataFrame(test_x1)).mode().iloc[0,0]
    if(col=="petition_primary_target_type"):
        test_x1 = le.fit_transform(test_x1)
    test_data_prepro = pd.concat([test_data_prepro,test_x1],axis=1)
    c=c+1
test_data_prepro.to_csv(data_dir+"validation_prepro.csv")
pd.DataFrame(test_x1).to_csv(data_dir+"validation_9.csv")
test_data_prepro = pd.read_csv(data_dir+"validation_prepro.csv")
test_data_prepro.columns = X_test.columns
y_pred = model.predict(test_data_prepro)
pd.DataFrame(y_pred).to_csv(data_dir+"outputclass.csv")
