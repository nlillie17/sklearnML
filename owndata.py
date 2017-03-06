#Conjunction with Austin, Max
#
# owndata.py
#
#

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd
from sklearn import tree
from sklearn import ensemble

''' data is from owndata.csv, data pylled from Fangraphs.
This dataset should enable us to predict if an MLB player
was an all-star based on certain season stats. '''

df = pd.read_csv('owndata.csv', header=0)
df.head()
df.info()

df = df.drop('Name', axis=1)
df = df.drop('Team', axis=1)
df = df.drop ('playerid', axis=1)

df = df.dropna()
print (df)

''' First experiment: using ML to predict if someone is an All-star guesser. 
We test 20 players with unknown All-Star status, and then use ML
to predict whether or not they were All-Stars'''

X_data = df.drop('All-Star', axis=1).values        # everything except the 'All-Star' column
y_data = df[ 'All-Star' ].values     

''' Now we remove the 20 unknown players'''

X_data_full = X_data[20:,:]
y_data_full = y_data[20:]

''' Scramble data'''

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]

X_test = X_data_full[0:20,0:19]              # the final testing data
X_train = X_data_full[20:,0:19]              # the training data
y_test = y_data_full[0:20]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[20:]                  # the training outputs/labels (known)

''' Changing the column weights to better reflect player value''' 

X_data[:,18] *= 550   #WAR is the most all econompassing stat of value so I think it'll have the biggest impact on all-star status
X_data[:,2] *= 200   #Everyone likes HRs, especially the fans voting for all-stars
X_data[:,9] *= 1 #Babip is a proxy for luck and only a little important

print('\n')
print("+++ start of DT analysis +++\n")

max_depth=6
#

optimal_depth0 = 0
best_score = 0

''' Gets optimal size tree to use for prediction'''

for n in range(1,max_depth+1):

    dtree = tree.DecisionTreeClassifier(max_depth=n)
    avg_score = 0
        
    for i in range(10):  # run at least 10 times.... take the average testing score
        #
        # split into our cross-validation sets...
        #
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        dtree = dtree.fit(cv_data_train, cv_target_train) 
        #print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
        #print("CV testing-data score:", dtree.score(cv_data_test,cv_target_test))
        avg_score += dtree.score(cv_data_test,cv_target_test)
    
    avg_score = avg_score / 10

    print("Average DT testing score at depth ++",n,"++ was: ", avg_score)

    if best_score < avg_score:
        best_score = avg_score
        optimal_depth0 = n

print("---------------------")
print("Optimal DT depth was ", optimal_depth0, "with a score of ", best_score, "\n")


''' Re-running the decision tree with the optimal depth'''

dtree = tree.DecisionTreeClassifier(max_depth=optimal_depth0)
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 
dtree = dtree.fit(cv_data_train, cv_target_train) 
score = dtree.score(cv_data_test,cv_target_test)


print("DT score was ", score)
print("Feature importances:", dtree.feature_importances_)
print("Predicted outputs are")
print(dtree.predict(X_test))
print("and the actual outcomes are")
print(y_test, "\n")

feature_names = df.columns.values
feature_names2 = []
for i in range(len(feature_names)):
    if feature_names[i] != "All-Star":
        feature_names2.append(feature_names[i])


target_names = ['0','1'] 
tree.export_graphviz(dtree, out_file='AS Tree' + str(max_depth) + '.dot',   # constructed filename!
                            feature_names=feature_names2,  filled=True, rotate=False, # LR vs UD
                            class_names=target_names, leaves_parallel=True) 

''' cross-validation step to get RT parameters (max_depth and n_estimators)'''

max_depth=6
#

optimal_depth = 0
best_score = 0

for n in range(1,max_depth+1):

    rforest = ensemble.RandomForestClassifier(max_depth=n, n_estimators=1000)
    avg_score = 0
        
    for i in range(10):  # run at least 10 times.... take the average testing score
        #
        # split into our cross-validation sets...
        #
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        rforest = rforest.fit(cv_data_train, cv_target_train)
        #print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
        #print("CV testing-data score:", dtree.score(cv_data_test,cv_target_test))
        avg_score += rforest.score(cv_data_test,cv_target_test)
    
    avg_score = avg_score / 10

    print("Average RT testing score at depth ++",n,"++ was: ", avg_score)

    if best_score < avg_score:
        best_score = avg_score
        optimal_depth = n

print("---------------------")
print("Optimal RT depth was ", optimal_depth, "with a score of ", best_score, "\n")



X_test = X_data_full[0:20,0:19]              # the final testing data
X_train = X_data_full[20:,0:19]              # the training data
y_test = y_data_full[0:20]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[20:]

max_depth = optimal_depth
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2)
rforest = ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=1000)

rforest = rforest.fit(X_train, y_train) 
score = rforest.score(cv_data_test,cv_target_test)
print("RT test score:", score)
print("RT feature importances:", rforest.feature_importances_) 

q = rforest.feature_importances_
print("The predicted outputs are")
print(rforest.predict(X_test),"\n")

print("and the actual labels are")
print(y_test)


df = df.astype(int)

''' Experiment two: use predicted All-Star Values (previously the dependent variable) to now predict
a player's war (new dependent variable) '''

df = pd.read_csv('owndata.csv', header=0)
df.head()
df.info()

df = df.drop('Name', axis=1)
df = df.drop('Team', axis=1)
df = df.drop ('playerid', axis=1)
df['WAR'].fillna(100, inplace=True)
df = df.sort('WAR', ascending=False)
df['WAR'] = df['WAR'].astype(int)

count = 0
for i in range(len(df['WAR'])):
    if df['WAR'][i] == 100:
        count +=1

X_data = df.drop('WAR', axis=1).values        # everything except the 'WAR' column
y_data = df[ 'WAR' ].values      # also addressable by column name(s)

#
# removing 20
#

X_data_full = X_data[count:,:]
y_data_full = y_data[count:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]

X_test = X_data_full[0:count,0:19]              # the final testing data
X_train = X_data_full[count:,0:19]              # the training data
y_test = y_data_full[0:count]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[count:]                  # the training outputs/labels (known)

cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
    cross_validation.train_test_split(X_train, y_train, test_size=0.2)
rforest = ensemble.RandomForestClassifier(max_depth=optimal_depth, n_estimators=1000)

rforest = rforest.fit(X_train, y_train) 
score = rforest.score(cv_data_test,cv_target_test)
print("Inputed test score:", score)
print("Inputed feature importances:", rforest.feature_importances_) 

predicted_values =  rforest.predict(X_test)
print("The predicted outputs are")
print(predicted_values,"\n")

print("and the actual labels are")
print(y_test)

print(df)

'''Update old data frame with inputes for WAR and see if it predicts All-Stars better    
'''   

for i in range(len(df['WAR'])):
    if df['WAR'][i] == 100:
        df['WAR'][i] = predicted_values[0]
        predicted_values = predicted_values[1:]

'''extract the underlying data with the All-star attribute:
'''

df = df.sort('All-Star', ascending=True)

print(df)
X_data = df.drop('All-Star', axis=1).values        # everything except the 'survival' column
y_data = df[ 'All-Star' ].values      # also addressable by column name(s)

#
# you can take away the top 20 players
#
X_data_full = X_data[20:,:]
y_data_full = y_data[20:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]


X_test = X_data_full[0:20,0:19]              # the final testing data
X_train = X_data_full[20:,0:19]              # the training data
y_test = y_data_full[0:20]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[20:]                  # the training outputs/labels (known)


dtree = tree.DecisionTreeClassifier(max_depth=optimal_depth0)
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 
dtree = dtree.fit(cv_data_train, cv_target_train) 
score = dtree.score(cv_data_test,cv_target_test)



print("Revised DT score with inputed ages and a depth of ",optimal_depth0, "was ", score)
print("Feature importances:", dtree.feature_importances_)
print("Predicted outputs are")
print(dtree.predict(X_test))
print("and the actual outcomes are")
print(y_test, "\n")

feature_names = df.columns.values
feature_names2 = []
for i in range(len(feature_names)):
    if feature_names[i] != "All-Star":
        feature_names2.append(feature_names[i])

for i in range(len(df['All-Star'])):
    if df['All-Star'][i] == -1:
        df['All-Star'][i] = dtree.predict(X_test[i])[0]
        

target_names = ['0','1'] 
tree.export_graphviz(dtree, out_file='AS Proj Tree' + str(max_depth) + '.dot',   # constructed filename!
                            feature_names=feature_names2,  filled=True, rotate=False, # LR vs UD
                            class_names=target_names, leaves_parallel=True) 

print (q)

'''
Problem 2:

Comments on data collection: We got the data from Fangraphs (fangraphs.com leaderboard export data), and manually entered all-star
    values for 180 of the 200 players based on espn's 2016 all-star roster. The first test was to predict if a player was an all-star. 
    Second experiment was to predict the WARs for players who had missing data.
    Then, we used the newly predicted wars as inputes and re-ran the first experiment.
 
1) The best DT model had a test-set accuracy of around 93%, but the average was around 75% 
2) The best RT model had a test-set accuracy of around 92%, and the averages were in the high 80%s.
3) Both DT screenshots and tree code is included in zip file. First tree is AS Tree10, second tree is AS Proj Tree6. 
The proj tree has projected WAR values in place for the 20 values that I left null. 
Attributes that were very important for predictive purposes were the RBIs in the projected tree while slugging was important
    in the actual model.
4)Feature Importances were the following:
  0.0442165   0.0626044   0.03637081  0.07721285  0.07871617  0.06656345
  0.02237428  0.02523101  0.02453182  0.02058898  0.04018957  0.0248628
  0.08117934  0.05041061  0.0703948   0.04298953  0.05056853  0.06854307
  0.1124515

  Column labels were: G  PA  HR  R   RBI SB  BB%, K%  ISO BABIP   AVG OBP SLG wOBA    wRC+    BsR Off Def WAR


5)      (0 = no and 1 = yes)
        The predicted all star outputs were
        0 0 0 1 0 0 1 0 1 1 0 0 0 1 0 1 0 0 0 1

        and the actual labels were 
        0 1 0 1 0 1 0 0 1 1 0 0 0 1 0 0 1 0 0 1
       
6) Because of perhaps the fan vote, the really all encompassing Sabermetric stats such as WAR aren't as important to predict All-star. 
I think it should because it is a better indicator of player value. Maybe if we made sabermetrics easier for the casual fan to understand
we would see more of a link between war and all-star status.

'''
