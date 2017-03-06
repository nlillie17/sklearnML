#Conjunction with Austin, Max
#
# titanic5.py
#
#

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd
from sklearn import tree
from sklearn import ensemble

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic5.csv', header=0)
df.head()
df.info()

# let's drop columns with too few values or that won't be meaningful
df = df.drop('name', axis=1)
df = df.drop('ticket', axis=1)
df = df.drop('fare', axis=1)
df = df.drop('cabin', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('home.dest', axis=1)



# let's drop all of the rows with missing data:
df = df.dropna()


# You'll need conversion to numeric datatypes for all input columns
#   Here's one example
#
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column


# We'll stick with numpy - here's the conversion to a numpy array

# extract the underlying data with the values attribute:
X_data = df.drop('survived', axis=1).values        # everything except the 'survival' column
y_data = df[ 'survived' ].values      # also addressable by column name(s)

#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
X_data_full = X_data[42:,:]
y_data_full = y_data[42:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]


X_test = X_data_full[0:42,0:12]              # the final testing data
X_train = X_data_full[42:,0:12]              # the training data
y_test = y_data_full[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[42:]                  # the training outputs/labels (known)

# feature engineering...
X_data[:,0] *= 550   #class is worth the most, rich people more likely to get saved
X_data[:,1] *= 430   #sex is worth a lot, the second most
X_data[:,2] *= 25  #relatively importnat
X_data[:,3] *= 20   #sibs important
X_data[:,4] *= 1   #parents 

print('\n')
print("+++ start of DT analysis +++\n")
#
# cross-validation to determine the Decision Tree's parameter (to find max_depth)
#
max_depth=10
#

optimal_depth0 = 0
best_score = 0

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



#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#


#rerun with optimal depth
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
    if feature_names[i] != "survived":
        feature_names2.append(feature_names[i])


target_names = ['0','1'] 
tree.export_graphviz(dtree, out_file='tree' + str(max_depth) + '.dot',   # constructed filename!
                            feature_names=feature_names2,  filled=True, rotate=False, # LR vs UD
                            class_names=target_names, leaves_parallel=True) 


#
# cross-validation to determine the Random Forest's parameters (max_depth and n_estimators)
#
#
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



X_test = X_data_full[0:42,0:12]              # the final testing data
X_train = X_data_full[42:,0:12]              # the training data
y_test = y_data_full[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[42:]

max_depth = optimal_depth
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2)
rforest = ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=1000)

rforest = rforest.fit(X_train, y_train) 
score = rforest.score(cv_data_test,cv_target_test)
print("RT test score:", score)
print("RT feature importances:", rforest.feature_importances_) 

print("The predicted outputs are")
print(rforest.predict(X_test),"\n")

print("and the actual labels are")
print(y_test)



#####inputing missing ages

df = pd.read_csv('titanic5.csv', header=0)
df.head()
df.info()

# let's drop columns with too few values or that won't be meaningful
df = df.drop('name', axis=1)
df = df.drop('ticket', axis=1)
df = df.drop('fare', axis=1)
df = df.drop('cabin', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('home.dest', axis=1)
df = df[np.isfinite(df['pclass'])]
df = df[np.isfinite(df['survived'])]
df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column
df = df[np.isfinite(df['sex'])]
df = df[np.isfinite(df['sibsp'])]
df = df[np.isfinite(df['parch'])]

#replace all NaN ages with -1
df['age'].fillna(-1, inplace=True)
df = df.sort('age', ascending=True)
df['age'] = df['age'].astype(int)

count = 0
for i in range(len(df['age'])):
    if df['age'][i] == -1:
        count +=1


# extract the underlying data with the values attribute:
X_data = df.drop('age', axis=1).values        # everything except the 'age' column
y_data = df[ 'age' ].values      # also addressable by column name(s)

#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
X_data_full = X_data[count:,:]
y_data_full = y_data[count:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]


X_test = X_data_full[0:count,0:5]              # the final testing data
X_train = X_data_full[count:,0:5]              # the training data
y_test = y_data_full[0:count]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[count:]                  # the training outputs/labels (known)

# feature engineering...
# X_data[:,0] *= 1  #class
X_data[:,1] *= 500   #survived
# X_data[:,2] *= 1  #sex
# X_data[:,3] *= 1   #sibs
# X_data[:,4] *= 1   #parents 

# instead of dropping all na, drop only non-age na's 
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


#now replace old df with new inputes for age, test to see if better predicts survivability       
for i in range(len(df['age'])):
    if df['age'][i] == -1:
        df['age'][i] = predicted_values[0]
        predicted_values = predicted_values[1:]

# extract the underlying data with the values attribute:
X_data = df.drop('survived', axis=1).values        # everything except the 'survival' column
y_data = df[ 'survived' ].values      # also addressable by column name(s)

#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
X_data_full = X_data[42:,:]
y_data_full = y_data[42:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]


X_test = X_data_full[0:42,0:12]              # the final testing data
X_train = X_data_full[42:,0:12]              # the training data
y_test = y_data_full[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[42:]                  # the training outputs/labels (known)

# feature engineering...
X_data[:,0] *= 550   #class is worth the most, rich people more likely to get saved
X_data[:,1] *= 430   #sex is worth a lot, the second most
X_data[:,2] *= 25  #relatively importnat
X_data[:,3] *= 20   #sibs important
X_data[:,4] *= 1   #parents 

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



# print("##-----------------------------------------##")
# cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
#     cross_validation.train_test_split(X_train, y_train, test_size=0.2)
# rforest = ensemble.RandomForestRegressor(max_depth=optimal_depth, n_estimators=1000)

# rforest = rforest.fit(X_train, y_train) 
# score = rforest.score(cv_data_test,cv_target_test)
# print("Inputed test score:", score)
# print("Inputed feature importances:", rforest.feature_importances_) 

# predicted_values =  rforest.predict(X_test)
# print("The predicted outputs are")
# print(predicted_values,"\n")

# print("and the actual labels are")
# print(y_test)

"""
Comments and results for Problem 2:

1) The best DT model had a test-set accuracy of 83.9%. 
2) The best RT model had a test-set accuracy of 83.4%
3) I also submitted the DT image. The first layer is a gender filter. If male, then 
is 'True' and goes to the left. If female, then is 'False' and goes to right. The 
second layer has to do with the class of the person. The male node sees if the class 
is <= 1.5 while the female node sees if the class is <= 2.5. For instance, the first line
of data in titanic5 is David Barton. The DT would thus evaluate as True, False...
4) The feature_importances_ for the RT model was:
    [0.18899425  0.49046108  0.18904908  0.06417804  0.06731754]
5)
        The predicted outputs are
        [21 18 30 21 18 30  1 28 21 29 25 43 45 39 25 21  4  0 35 24 50 35 22 32  9
        49 21 47 50 28 21 37 21 50 24 21 21 36 21 21 21 22  9 22 43 21 30 24 21 21
        18 49 21 21 21 21 21 30 39 21  9 21  2 49 47 43 36 35 32 18 18 30  4 47 24
        21 21 21 45 21 37 35 47 21 21 21 18 21 21 21 18 18 45 32 21 43 21 21 21 22
        18 18 32 21 32 24 39 35 21 50 43 21 29 18 47 35 55 22 22 24 32 21 21 21 21
        28 50 35 35 24 21 13 50  2 22 21 35 21 22 21 24 21 21 35 35 18 47 36 35 22
        21 39 21 45 21 21 21 24 47 21 21 13 21 30 47 21 21 21 22 22 21  0 35 21 21
        39 30 28 37 47  1 50 14 21 21 47 36 24 18 29 39 30 21 35 21 18 21 21 22 21
        21 50 50 47 21 13 32 21 47 21 21 36  9  0 21 35 18 22 40 21 21 21 50 32 21
        2 32 47 22 35 29 18 45 47 50 47 21  1 30 24 47 18 21 21 21 45 35 37 30 21
        50 18 21 39 21 21 47 47 21 35 21 35  2 21 18 21 21 21 21 18 32  0 18 35 21
        36 21 22 32 35 21 21  0 37 18  9 24 21  0 24 21] 

        and the actual labels are
        [17 21 17 22 19 42 31 20 25 34 18 31 42 54 24 33 24  3 29 48 18 30 18 27 38
        31 21 47 27 42 20 50 39 44 18 57 40 41 29 21 26 36 10 23 31 24 55 54 30 22
        30 38 42 28 21 21 32 34 41 27  0 29  4 24 22 19 23 47 34 29 22 27  6 40 41
        33 41 22 35 41 21 17 41 27 41 30 18 26 27 28 13 33 51 25 14 30 19 36 16 56
        29 42 27 20 24 24 48 35 21 63 42 22 34 25 40 52 59 47 45 29 44 19 40 24 16
        24 41 33 30 21 21  9 67 10 22 36 30 28 15 47  2 23 24 30 24 20 60 10 54 27
        36 52 28 26 17 19 27 33 29 50 28  4 23 24 54 34 33 26 16 43 23  1 44 47 30
        39 34 29 53 30 16 42 11 19 25 28 24  8 24 50 39 17 32 19 35 21 70 57 16 32
        25 36 36 42 25  5 29 27 46 30 26 13  0  5 39 18 37 45 40 28 27 28 37 27 22
        4 32 36 22 30 21 34 32 48 47 61 19  0 36 15 42 50 24 38 41 34 22 52 19 18
        64 25 18 45 24 24 33 46 18 24 35 37  7 25 31 18 33 23 32 20 31  3  2 53 20
        49 25 18 20 60 21 18 26 24 22 16  8 28  9 21 16]

6) Tree:

digraph Tree {
node [shape=box, style="filled", color="black"] ;
graph [ranksep=equally, splines=polyline] ;
0 [label="sex <= 0.5\ngini = 0.475\nsamples = 747\nvalue = [457, 290]\nclass = 0", fillcolor="#e581395d"] ;
1 [label="pclass <= 1.5\ngini = 0.3086\nsamples = 472\nvalue = [382, 90]\nclass = 0", fillcolor="#e58139c3"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
2 [label="age <= 54.5\ngini = 0.4469\nsamples = 89\nvalue = [59, 30]\nclass = 0", fillcolor="#e581397d"] ;
1 -> 2 ;
3 [label="age <= 17.5\ngini = 0.4892\nsamples = 68\nvalue = [39, 29]\nclass = 0", fillcolor="#e5813941"] ;
2 -> 3 ;
4 [label="gini = 0.0\nsamples = 3\nvalue = [0, 3]\nclass = 1", fillcolor="#399de5ff"] ;
3 -> 4 ;
5 [label="age <= 47.5\ngini = 0.48\nsamples = 65\nvalue = [39, 26]\nclass = 0", fillcolor="#e5813955"] ;
3 -> 5 ;
6 [label="gini = 0.4561\nsamples = 54\nvalue = [35, 19]\nclass = 0", fillcolor="#e5813975"] ;
5 -> 6 ;
7 [label="gini = 0.4628\nsamples = 11\nvalue = [4, 7]\nclass = 1", fillcolor="#399de56d"] ;
5 -> 7 ;
8 [label="age <= 56.5\ngini = 0.0907\nsamples = 21\nvalue = [20, 1]\nclass = 0", fillcolor="#e58139f2"] ;
2 -> 8 ;
9 [label="age <= 55.5\ngini = 0.32\nsamples = 5\nvalue = [4, 1]\nclass = 0", fillcolor="#e58139bf"] ;
8 -> 9 ;
10 [label="gini = 0.0\nsamples = 3\nvalue = [3, 0]\nclass = 0", fillcolor="#e58139ff"] ;
9 -> 10 ;
11 [label="gini = 0.5\nsamples = 2\nvalue = [1, 1]\nclass = 0", fillcolor="#e5813900"] ;
9 -> 11 ;
12 [label="gini = 0.0\nsamples = 16\nvalue = [16, 0]\nclass = 0", fillcolor="#e58139ff"] ;
8 -> 12 ;
13 [label="age <= 9.5\ngini = 0.2642\nsamples = 383\nvalue = [323, 60]\nclass = 0", fillcolor="#e58139d0"] ;
1 -> 13 ;
14 [label="sibsp <= 2.5\ngini = 0.4835\nsamples = 22\nvalue = [13, 9]\nclass = 0", fillcolor="#e581394e"] ;
13 -> 14 ;
15 [label="age <= 0.875\ngini = 0.375\nsamples = 12\nvalue = [3, 9]\nclass = 1", fillcolor="#399de5aa"] ;
14 -> 15 ;
16 [label="gini = 0.0\nsamples = 2\nvalue = [2, 0]\nclass = 0", fillcolor="#e58139ff"] ;
15 -> 16 ;
17 [label="gini = 0.18\nsamples = 10\nvalue = [1, 9]\nclass = 1", fillcolor="#399de5e3"] ;
15 -> 17 ;
18 [label="gini = 0.0\nsamples = 10\nvalue = [10, 0]\nclass = 0", fillcolor="#e58139ff"] ;
14 -> 18 ;
19 [label="age <= 32.5\ngini = 0.2426\nsamples = 361\nvalue = [310, 51]\nclass = 0", fillcolor="#e58139d5"] ;
13 -> 19 ;
20 [label="age <= 28.75\ngini = 0.2891\nsamples = 251\nvalue = [207, 44]\nclass = 0", fillcolor="#e58139c9"] ;
19 -> 20 ;
21 [label="gini = 0.2582\nsamples = 197\nvalue = [167, 30]\nclass = 0", fillcolor="#e58139d1"] ;
20 -> 21 ;
22 [label="gini = 0.3841\nsamples = 54\nvalue = [40, 14]\nclass = 0", fillcolor="#e58139a6"] ;
20 -> 22 ;
23 [label="age <= 36.25\ngini = 0.1192\nsamples = 110\nvalue = [103, 7]\nclass = 0", fillcolor="#e58139ee"] ;
19 -> 23 ;
24 [label="gini = 0.0\nsamples = 34\nvalue = [34, 0]\nclass = 0", fillcolor="#e58139ff"] ;
23 -> 24 ;
25 [label="gini = 0.1672\nsamples = 76\nvalue = [69, 7]\nclass = 0", fillcolor="#e58139e5"] ;
23 -> 25 ;
26 [label="pclass <= 2.5\ngini = 0.3967\nsamples = 275\nvalue = [75, 200]\nclass = 1", fillcolor="#399de59f"] ;
0 -> 26 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
27 [label="pclass <= 1.5\ngini = 0.1327\nsamples = 154\nvalue = [11, 143]\nclass = 1", fillcolor="#399de5eb"] ;
26 -> 27 ;
28 [label="age <= 35.5\ngini = 0.0476\nsamples = 82\nvalue = [2, 80]\nclass = 1", fillcolor="#399de5f9"] ;
27 -> 28 ;
29 [label="gini = 0.0\nsamples = 43\nvalue = [0, 43]\nclass = 1", fillcolor="#399de5ff"] ;
28 -> 29 ;
30 [label="age <= 37.0\ngini = 0.0973\nsamples = 39\nvalue = [2, 37]\nclass = 1", fillcolor="#399de5f1"] ;
28 -> 30 ;
31 [label="gini = 0.375\nsamples = 4\nvalue = [1, 3]\nclass = 1", fillcolor="#399de5aa"] ;
30 -> 31 ;
32 [label="gini = 0.0555\nsamples = 35\nvalue = [1, 34]\nclass = 1", fillcolor="#399de5f8"] ;
30 -> 32 ;
33 [label="age <= 55.5\ngini = 0.2188\nsamples = 72\nvalue = [9, 63]\nclass = 1", fillcolor="#399de5db"] ;
27 -> 33 ;
34 [label="parch <= 1.5\ngini = 0.18\nsamples = 70\nvalue = [7, 63]\nclass = 1", fillcolor="#399de5e3"] ;
33 -> 34 ;
35 [label="gini = 0.2188\nsamples = 56\nvalue = [7, 49]\nclass = 1", fillcolor="#399de5db"] ;
34 -> 35 ;
36 [label="gini = 0.0\nsamples = 14\nvalue = [0, 14]\nclass = 1", fillcolor="#399de5ff"] ;
34 -> 36 ;
37 [label="gini = 0.0\nsamples = 2\nvalue = [2, 0]\nclass = 0", fillcolor="#e58139ff"] ;
33 -> 37 ;
38 [label="age <= 0.875\ngini = 0.4983\nsamples = 121\nvalue = [64, 57]\nclass = 0", fillcolor="#e581391c"] ;
26 -> 38 ;
39 [label="gini = 0.0\nsamples = 3\nvalue = [0, 3]\nclass = 1", fillcolor="#399de5ff"] ;
38 -> 39 ;
40 [label="sibsp <= 1.5\ngini = 0.4964\nsamples = 118\nvalue = [64, 54]\nclass = 0", fillcolor="#e5813928"] ;
38 -> 40 ;
41 [label="age <= 27.5\ngini = 0.5\nsamples = 101\nvalue = [51, 50]\nclass = 0", fillcolor="#e5813905"] ;
40 -> 41 ;
42 [label="gini = 0.4931\nsamples = 68\nvalue = [30, 38]\nclass = 1", fillcolor="#399de536"] ;
41 -> 42 ;
43 [label="gini = 0.4628\nsamples = 33\nvalue = [21, 12]\nclass = 0", fillcolor="#e581396d"] ;
41 -> 43 ;
44 [label="age <= 5.5\ngini = 0.3599\nsamples = 17\nvalue = [13, 4]\nclass = 0", fillcolor="#e58139b1"] ;
40 -> 44 ;
45 [label="gini = 0.5\nsamples = 4\nvalue = [2, 2]\nclass = 0", fillcolor="#e5813900"] ;
44 -> 45 ;
46 [label="gini = 0.2604\nsamples = 13\nvalue = [11, 2]\nclass = 0", fillcolor="#e58139d1"] ;
44 -> 46 ;
{rank=same ; 0} ;
{rank=same ; 1; 26} ;
{rank=same ; 2; 13; 27; 38} ;
{rank=same ; 3; 8; 14; 19; 28; 33; 40} ;
{rank=same ; 5; 9; 15; 20; 23; 30; 34; 41; 44} ;
{rank=same ; 4; 6; 7; 10; 11; 12; 16; 17; 18; 21; 22; 24; 25; 29; 31; 32; 35; 36; 37; 39; 42; 43; 45; 46} ;
}







"""