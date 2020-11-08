import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

training=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
tdata=pd.read_csv('gender_submission.csv')




from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# print(training.columns)
# print(test.columns)
# print(training.info)
# print(training.info())
# print(training.describe())
# train_cat=training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
# train_num=training[['Age','SibSp','Parch','Fare']]
# print(train_num)
# for i in train_num.columns:
#     plt.hist(train_num[i])
#     plt.title(i)
#     plt.show()
# print(train_num.corr())
# sns.heatmap(train_num.corr())
# plt.show()
# print(pd.pivot_table(training,index="Survived",values=['Age','SibSp','Parch','Fare']))
# for i in train_cat.columns:
#     sns.barplot(train_cat[i].value_counts().index,train_cat[i].value_counts()).set_title(i)
#     plt.show()

# print(pd.pivot_table(training,index="Survived",columns='Pclass',values='Ticket',aggfunc='count'))
# print(pd.pivot_table(training,index="Survived",columns='Sex',values='Ticket',aggfunc='count'))
# print(pd.pivot_table(training,index="Survived",columns='Embarked',values='Ticket',aggfunc='count'))

training['Age']=training['Age'].fillna(29.69)
sex=list(training['Sex'])
embark=list(training['Embarked'])
cl=list(training['Pclass'])
training['Male']=[int(sex[i]=='male') for i in range(0,len(sex))]
training['Female']=[int(sex[i]=='female') for i in range(0,len(sex))]
training['Embarkeds']=[int(embark[i]=='S') for i in range(0,len(sex))]
training['Embarkedc']=[int(embark[i]=='C') for i in range(0,len(sex))]
training['Embarkedq']=[int(embark[i]=='Q') for i in range(0,len(sex))]
training['class1']=[int(cl[i]==1) for i in range(0,len(sex))]
training['class2']=[int(cl[i]==2) for i in range(0,len(sex))]
training['class3']=[int(cl[i]==3) for i in range(0,len(sex))]
train=training.drop(columns=['Sex', 'Embarked','Name','PassengerId','Ticket','Cabin','Pclass'])
# train.apply(f, axis=1)
# training['']=training['Age'].fillna(29.69)
# # train['Fare']=(train['Fare']-32)/49.69
# print(train.describe())
# print(train)
ytrain=training['Survived']
xtrain=np.array(train[['Age','SibSp','Parch','Fare','Male','Female','Embarkeds','Embarkedc','Embarkedq','class1','class2','class3']])









test['Age']=test['Age'].fillna(29.69)
test['Fare']=test['Fare'].fillna(35.627)
sex=list(test['Sex'])
embark=list(test['Embarked'])
cl=list(test['Pclass'])
test['Male']=[int(sex[i]=='male') for i in range(0,len(sex))]
test['Female']=[int(sex[i]=='female') for i in range(0,len(sex))]
test['Embarkeds']=[int(embark[i]=='S') for i in range(0,len(sex))]
test['Embarkedc']=[int(embark[i]=='C') for i in range(0,len(sex))]
test['Embarkedq']=[int(embark[i]=='Q') for i in range(0,len(sex))]
test['class1']=[int(cl[i]==1) for i in range(0,len(sex))]
test['class2']=[int(cl[i]==2) for i in range(0,len(sex))]
test['class3']=[int(cl[i]==3) for i in range(0,len(sex))]
testing=test.drop(columns=['Sex', 'Embarked','Name','PassengerId','Ticket','Cabin','Pclass'])
# train.apply(f, axis=1)
# training['']=training['Age'].fillna(29.69)
# # train['Fare']=(train['Fare']-32)/49.69
# print(train.describe())
# print(train)
# ytest=testing['Survived']

xtest=testing[['Age','SibSp','Parch','Fare','Male','Female','Embarkeds','Embarkedc','Embarkedq','class1','class2','class3']]
# print(xtest.describe())







# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)



clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf_svm.fit(xtrain,ytrain)
# Pipeline(steps=[('standardscaler', StandardScaler()),('svc', SVC(gamma='auto'))])


a=accuracy_score(ytrain,clf_svm.predict(xtrain))
print(" \n  ")
print('svm train acc: ',a)
# b=accuracy_score(ytest,clf_svm.predict(xtest))
# print('svm test acc: ',b)
# print(" \n  ")







clf_dtree = tree.DecisionTreeClassifier()
clf_dtree.fit(xtrain,ytrain)
a=accuracy_score(ytrain,clf_dtree.predict(xtrain))
print('decision tree train acc: ',a)
# b=accuracy_score(ytest,clf_dtree.predict(xtest))
# print('decision tree test acc: ',b)
# print(" \n  ")




clf_xgboost = GradientBoostingClassifier(random_state=0)
clf_xgboost.fit(xtrain, ytrain)
a=accuracy_score(ytrain,clf_xgboost.predict(xtrain))
print('xgboost train acc: ',a)
# b=accuracy_score(ytest,clf_xgboost.predict(xtest))
# print('xgboost test acc: ',b)
# print(" \n  ")
p=clf_xgboost.predict(xtest)
tdata["Survived"]=p


clf_forest = RandomForestClassifier(max_depth=2, random_state=0)
clf_forest.fit(xtrain, ytrain)
a=accuracy_score(ytrain,clf_forest.predict(xtrain))
print('random forest train acc: ',a)
# b=accuracy_score(ytest,clf_forest.predict(xtest))
# print('forest test acc: ',b)
# print("  \n ")

# print(p)
# print(tdata)
# print("\n")
# print(tdata.describe())
td=tdata[['PassengerId','Survived']]
td.to_csv('sub3.csv')
