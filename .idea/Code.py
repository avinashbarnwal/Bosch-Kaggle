import h2o
import numpy as np

# Start H2O on your local machine
h2o.init ( ip='localhost', port=54321, nthreads=-1, max_mem_size='25g' )
# Import the train_numeric
train_numeric = h2o.import_file ( path="/Users/avinashbarnwal/Desktop/Kaggle/Bosch/train_numeric.csv" )
# train_categorical = h2o.import_file(path = "/Users/avinashbarnwal/Desktop/Kaggle/Bosch/train_categorical.csv")
# train_date = h2o.import_file(path = "/Users/avinashbarnwal/Desktop/Kaggle/Bosch/train_date.csv")


# train_numeric.describe()
train_numeric["Response"] = train_numeric["Response"].asfactor ( )

# train = train_numeric.cbind(train_categorical)
# train=  train.cbind(train_date)


train_numeric_name = train_numeric.col_names
# train_categorical_name = train_categorical.col_names
# train_date_name = train_date.col_names

# Yielding Colnames of train_numeric
t = train_numeric.shape
# Yielding Dimension of train_numeric
removelist_train_numeric = [0, (t[1] - 1)]
# Removing Id and response
train_numeric_name = [v for i, v in enumerate ( train_numeric_name ) if i not in removelist_train_numeric]

# print train_numeric_name

# removelist_train_categorical = [0]
# train_categorical_name   =     [v for i, v in enumerate(train_categorical_name) if i not in removelist_train_categorical]

# removelist_train_date = [0]
# train_date_name = [v for i, v in enumerate(train_date_name) if i not in removelist_train_date]

# Concatenate the arrays
# train_name =sum([train_numeric_name, train_categorical_name, train_date_name],[])





# Train GBM model

gbm_model = h2o.estimators.gbm.H2OGradientBoostingEstimator ( ntrees=100,
                                                              max_depth=5,
                                                              learn_rate=0.1,
                                                              sample_rate=0.8,
                                                              col_sample_rate=0.8,
                                                              seed=2016,
                                                              nfolds=5,
                                                              fold_assignment="Stratified",
                                                              keep_cross_validation_predictions='TRUE',
                                                              keep_cross_validation_fold_assignment='TRUE',
                                                              score_tree_interval=20,
                                                              stopping_rounds=10,
                                                              stopping_metric="AUC",
                                                              stopping_tolerance=0.01 )

gbm_model.train ( x=train_numeric_name, y="Response", training_frame=train_numeric )


importance = gbm_model.varimp ( )
# print importance


# importance <- as.dataframe(importance)
# importance <- importance[scaled_importance>0]
