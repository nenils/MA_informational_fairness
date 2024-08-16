import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
#from imblearn.over_sampling import RandomOversampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
#from fairlearn.adversarial import AdversarialFairnessClassifier
import pickle
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
#from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
#from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
#from aif360.algorithms.preprocessing import Reweighing
#from aif360.datasets import BinaryLabelDataset
#from aif360.metrics import BinaryLabelDatasetMetric
#from aif360.metrics import ClassificationMetric


def load_and_preprocess_data(path):
    path = str(path)
    dataset = pd.read_csv(path)
    
    dataset.dropna(subset=['LoanAmount','Loan_Amount_Term','Credit_History','Dependents','Self_Employed','Gender','Loan_ID'], inplace=True)
    dataset.drop(['Loan_ID'], axis=1, inplace=True)
    target = dataset["Loan_Status"]
    dataset.drop(['Loan_Status'], axis=1, inplace=True)
    datasetX = dataset.copy()
    numerical = datasetX.select_dtypes(include='number').columns.tolist()
    categorical = datasetX.select_dtypes(include='object').columns.tolist()

    
    dataset = datasetX.copy()
    dataset['Loan_Status'] = target

    return datasetX, dataset ,target, numerical, categorical

def do_it_like_numbers_do(datasetX,target,numerical, categorical):

    datasetX = pd.get_dummies(datasetX, columns=['Property_Area'])

    datasetX['Property_Area_Semiurban'].rename('Property_Area', inplace=True)

    #datasetX['Property_Area'] = datasetX['Property_Area'].astype('int64')
    categorical.remove('Property_Area')

    scaler = MinMaxScaler()
    datasetX[numerical] = scaler.fit_transform(datasetX[numerical])
   
    for col in categorical:
        encoder = LabelEncoder()
        datasetX[col] = encoder.fit_transform(datasetX[col])
    target = encoder.fit_transform(target)

    datasetX['Credit_History'] = datasetX['Credit_History'].astype('int64')
    dataset_num_headers = datasetX.copy()
    datasetX1 = np.asarray(datasetX).astype(np.float32)
    
    #target  = np.asarray(target).astype('int64')
    sensitive_attributes = ['Gender','Property_Area']
    Z = datasetX[sensitive_attributes]
    x_train_encoded, x_test_encoded, y_train, y_test, Z_train, Z_test = train_test_split(datasetX,
                                                        target,Z, 
                                                        test_size=0.2,
                                                        random_state=123,
                                                        stratify=target)
    
    x_train_encoded = pd.DataFrame(x_train_encoded, columns=datasetX.columns)
    x_test_encoded = pd.DataFrame(x_test_encoded, columns=datasetX.columns)
    y_train = pd.Series(y_train)
    y_test = pd.DataFrame(y_test, columns=['Loan_Status'],index=x_test_encoded.index)
    Z_train = pd.DataFrame(Z_train, columns=sensitive_attributes)
    Z_test = pd.DataFrame(Z_test, columns=sensitive_attributes)


    return x_train_encoded, x_test_encoded, y_train, y_test, Z_train, Z_test, datasetX1, target,Z, dataset_num_headers, scaler, encoder


def build_pipeline(numerical, categorical):

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)])
    clf = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000000))])
    tree = Pipeline(steps=[('preprocessor', transformations),
                          ('tree', DecisionTreeClassifier())])
    Naive = Pipeline(steps=[('preprocessor', transformations),
                            ('NB', GaussianNB())])
    lr = Pipeline(steps=[('preprocessor', transformations),
                            ('LR', LogisticRegression())])
    models = {'MLP': clf, 'RandomForest': tree, 'NaiveBayes': Naive, 'LogisticRegression': lr}
    model_list = [clf, tree, Naive, lr]
    return models, model_list, transformations
    
def nn_models_build():
    predictor_model = tf.keras.Sequential([tf.keras.layers.Dense(50, activation='relu'), tf.keras.layers.Dense(1)])
    adversary_model = tf.keras.Sequential([tf.keras.layers.Dense(3, activation='relu'),tf.keras.layers.Dense(1)])
    
    predictor_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    adversary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    mitigator = AdversarialFairnessClassifier(
    backend="torch",
    predictor_model=[50, "leaky_relu"],
    adversary_model=[3, "leaky_relu"],
    batch_size=2 ** 8,
    progress_updates=0.5,
    random_state=123,
)
    

    return mitigator

def train_model_nn(model, x_train, y_train, x_test, Z_train):
    predictions = {} 
    accuracys = {} 
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    #filename = str(key)+'.keras'
    #value.save(filename)
    predictions['mitigator_model'] = prediction
    return predictions



def train_model(models, x_train, y_train, x_test, y_test):
    predictions = [] 
    accuracys = {} 
    for key, value in models.items():
        value.fit(x_train, y_train)
        prediction = value.predict(x_test)
        accuracy = value.score(x_test, y_test)
        filename = str(key)+'.sav'
        pickle.dump(value, open(filename, 'wb'))
        predictions.append(prediction)
        accuracys[key] = accuracy
    return predictions, accuracys


def check_class_imbalance(df, target):
    class_values = df[target].value_counts()
    print(class_values)
    
    if len(class_values) > 2:
        print("Multiclass classification")
        
        majority_class = class_values.idxmax()
        minority_class = [cls for cls in class_values.index if cls != majority_class][0]
        print("Majority Class is: ", majority_class)
        print("Minority Class is: ", minority_class)
        
        majority_count = class_values[majority_class]
        minority_count = class_values[minority_class]
        
        if (majority_count / minority_count) > 1.5:
            print("Class imbalance exists")
            return True
        else:
            print("Class imbalance does not exist")
            return False    
            
    elif len(class_values) == 2:
        print("Binary classification")
        majority_class = class_values.idxmax()
        minority_class = class_values.idxmin()
        print("Majority Class is: ", majority_class)
        print("Minority Class is: ", minority_class)
        
        majority_count = class_values[majority_class]
        minority_count = class_values[minority_class]
        
        if (majority_count / minority_count) > 1.5:
            print("Class imbalance exists")
            return True
        else:
            print("Class imbalance does not exist")
            return False
            
    else:
        print("No class imbalance")
        return False    


def fair_sampling(datasetX, target):
    
    dataset = datasetX.copy()
    dataset['target']= target
    sampling_target = datasetX['Gender']

    dataset.drop(['Gender'], axis=1, inplace=True)
    # Check class imbalance
    if check_class_imbalance:
        ros = RandomOverSampler(sampling_strategy='auto',random_state=0)
        datasetX_resampled, sampling_target_resampled  = ros.fit_resample(dataset, sampling_target)
        
        datasetX_resampled  = pd.concat([datasetX_resampled, sampling_target_resampled], axis=1)
        
        print(print('Resampled dataset shape %s' % Counter(datasetX_resampled)) )
        print(print('Resampled dataset shape %s' % Counter(sampling_target_resampled)))
        target_resampled = datasetX_resampled['target']
        
        datasetX_resampled.drop(['target'], axis=1, inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(datasetX_resampled,
                                target_resampled, 
                                test_size= 0.2,
                                random_state=123,
                                stratify=target_resampled)
        
        
    else:    
        print ("No class imbalance")
    
    return x_train, x_test, y_train, y_test, datasetX_resampled, target_resampled
    

def fair_model_build (x_train,y_train,x_test,y_test,Z_train,Z_test):
    pipeline = Pipeline(
        [   (
                "classifier",
                AdversarialFairnessClassifier(
                    backend="torch",
                    predictor_model=[50, "leaky_relu"],
                    adversary_model=[3, "leaky_relu"],
                    batch_size=2 ** 8,
                    random_state=123,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train,classifier__sensitive_features=Z_train)
    predictions = pipeline.predict(x_test)
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate
                 ,"precision": precision_score, "recall": recall_score, "f1": f1_score},
        y_true=y_test == 1,
        y_pred=predictions == 1,
        sensitive_features=Z_test,
    )
    print(mf.by_group)
    return pipeline, predictions, mf


def adversarial_model_build (dataset_num_headers,target,scaler,p,u):

    dataset_num_headers.replace({True: 1, False: 0}, inplace=True)
    
    dataset_num_headers['Loan_Status'] = target
    

    # Convert dataframe to structured dataset
    structured_dataset = StructuredDataset(df=dataset_num_headers, label_names=['Loan_Status'], protected_attribute_names=['Gender'])
    structured_dataset_df,dict = structured_dataset.convert_to_dataframe()

    ad = BinaryLabelDataset(favorable_label=1.,unfavorable_label=0.,df=structured_dataset_df,label_names=dict['label_names'],protected_attribute_names=dict['protected_attribute_names'])
    test, train = ad.split(num_or_size_splits=2)
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.fit_transform(test.features)

    index = train.feature_names.index("Gender")
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    sess1_ad = tf.Session()

    debiased_model_ad = AdversarialDebiasing(privileged_groups = p,
                            unprivileged_groups = u,
                            scope_name='debiased_classifier',
                            debias=True,
                            sess=sess1_ad)

    return debiased_model_ad

def aif360_fair_prepro(dataset, unprivileged_groups, privileged_groups):

    bi_dataset = BinaryLabelDataset(favorable_label=1,unfavorable_label=0,df=dataset,label_names=['Loan_Status'],protected_attribute_names='0')

    #optim_options = {
    #    "epsilon": 0.05,
    #    "clist": [0.99, 1.99, 2.99],
    #    "dlist": [.1, 0.05, 0]
    #}
    # --> Reference:  F. P. Calmon, D. Wei, B. Vinzamuri, K. Natesan Ramamurthy, and K. R. Varshney. “Optimized Pre-Processing for Discrimination Prevention.” Conference on Neural Information Processing Systems, 2017.
    opt = OptimPreproc(OptTools, unprivileged_groups, privileged_groups)
    #reweigh = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)


    #reweigh.fit(dataset)
    opt.fit(bi_dataset)


    #dataset_reweigh = reweigh.transform(dataset)
    dataset_opt = opt.transform(bi_dataset)

    return dataset_opt
# Example usage:
#datasetX, target, model, numerical, categorical, x_train, x_test, y_train, y_test = load_and_preprocess_data()
#models = build_pipeline(numerical, categorical)
#predictions = train_model(models, x_train, y_train)



# Distortion function from AIF360 library and the optim_preproc_helpers

def get_distortion_loan(vold, vnew):
    """Distortion function for the german dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Distortion cost
    distort = {}
    distort['Credit_History'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['Self_Employed'] = pd.DataFrame(
                        {1 :    [0., 1.],
                         0:    [2., 0.]},
                         index=['No', 'Yes'])
    distort['Education'] = pd.DataFrame(
                        {1:    [0., 1.],
                         0:    [2., 0.]},
                         index=['Not Graduate', 'Graduate'])
    distort['Married'] = pd.DataFrame(
                        {1 :    [0., 1.],
                         0:    [2., 0.]},
                         index=['No', 'Yes'])
    distort['Dependents'] = pd.DataFrame(
                            {'0':          [0., 1., 2., 3.],
                            '1':           [1., 0., 1., 0.],
                            '2':           [0., 1., 0., 1.],
                            '3+':          [3., 2., 1., 0.]},
                            index=['0', '1', '2','3+'])
    distort['Property_Area'] = pd.DataFrame(
                        {'Rural' :    [0., 1., 2.],
                         'Urban':    [1., 0., 1.],
                         'Semiurban':    [2., 1., 0.]},
                         index=['Rural', 'Urban', 'Semiurban'])
    distort['Gender'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost