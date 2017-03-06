# sklearnML
Using sklearn machine learning libraries to use DT and RFs for prediction (computer science at harvey mudd college).

##Purpose
The purpose of this project was to use machine learning libraries in python for predictive statistical modeling. The algorithims I focused on for this project were Decision Trees and Random Forests. The following are the descriptions of the  exercises that I did:

1) Using data (provided by class) about people aboard the titanic to predict whether they lived or survived.
  a) Initially I dropped people from the dataset if they had missing values but I also re-ran the model using inputed values for age. Essentially, instead of dropping individuals who didn't have a known age, I simply used age as the dependnet variable, using RFs to predict the value. Once I had that prediction, I plugged the age data in and re-ran the model.
2) Using Fan graphs data to predict whether a player would be an All-star. Similar to the above example, I first ran it dropping those players who didn't have a WAR stat (Wins above replacement - a sabermetric, 'catch-all' type statistic). For the next time I inputed the WARs using RFs. The I re-ran the model using the inputed WARs.

  *Note: it must be said the procedure for running the algorithims was identical between the two-data sets. The things that were changed between them involved data-cleaning and feature-engineering.

##Results

1) For the titanic data-set, the best DT model had a test-set accuracy of 84%. The best RF model had a test-set accuracy of 83%. The optimal size of the DT was 5. I found this by running cross validation that averaged the accuracy of 10 different models with a depth of i in range (0,max_depth). I set max_depth equal to 10. 
2) The best DT model had a test-set accuracy of around 93%, but the average was around 75%. The best RT model had a test-set accuracy of around 92%, and the averages were in the high 80%s. Slugging was found to be the most important feature.
  a) Interesting insight: Because of perhaps the fan vote, the really all encompassing Sabermetric stats such as WAR aren't as important to predict All-star. I think it should because it is a better indicator of player value. Maybe if we made sabermetrics easier for the casual fan to understand we would see more of a link between war and all-star status.
  

###Selected screenshots from attached files

* cross validation example to find the optimal size of the DT 
[cross validation](https://github.com/nlillie17/sklearnML/blob/master/cross.png)
* running the prediction with the optimal size
[prediction](https://github.com/nlillie17/sklearnML/blob/master/prediction.png)
* feature engineering
[feature engineering](https://github.com/nlillie17/sklearnML/blob/master/feature_engineering.png)
