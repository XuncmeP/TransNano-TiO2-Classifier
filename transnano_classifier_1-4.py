import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, auc,  f1_score, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import logging
from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler

def domain_applcability(X_train, X_test, y_train, y_test):
    
    names_train = []
    names_test = []
    
    train_arr = np.zeros(X_train.shape)
    test_arr = np.zeros(X_test.shape)   
    list_tr_delate = []    
    list_test_delate = []
    for i in range(X_train.shape[1]):
        for k in range(X_train.shape[0]):
            Ski = (X_train.iloc[k,i]-X_train.mean()[i])/X_train.std()[i]
            train_arr[k,i] = Ski
            
    for i in range(X_test.shape[1]):
        for k in range(X_test.shape[0]):
            test_s = (X_test.iloc[k,i]-X_train.mean()[i])/X_train.std()[i]
            test_arr[k,i] = test_s
    
    for _ in range(len(train_arr)):
        if max(train_arr[_]) > 3 and min(train_arr[_]) > 3:
            list_tr_delate.append(_)
        elif max(train_arr[_]) > 3 and min(train_arr[_]) < 3:
            new_S_train = np.mean(train_arr[_]) + 1.28 * np.std(train_arr[_])

            if new_S_train > 3:
                list_tr_delate.append(_)
        
    train_arr = np.delete(train_arr, list_tr_delate, 0)
        
    for _ in range(len(test_arr)):
        if max(test_arr[_]) > 3 and min(test_arr[_]) > 3:
            list_test_delate.append(_)
        elif max(test_arr[_]) > 3 and min(test_arr[_]) < 3:
            new_S_test = np.mean(test_arr[_]) + 1.28 * np.std(test_arr[_])

            if new_S_test > 3:
                list_test_delate.append(_)
            
    test_arr = np.delete(test_arr, list_test_delate, 0)
    print(list_tr_delate, list_test_delate)    
    
    names_train.append(list(X_train.index[list_tr_delate]))
    names_test.append(list(X_test.index[list_test_delate]))
    
    #X_train.drop(index=names_train[0], axis=0, inplace= True)
    #X_test.drop(index=names_test[0], axis=0, inplace = True)
    
    #y_train = y_train.drop(index=names_train[0], axis=0)
    #y_test = y_test.drop(index=names_test[0], axis=0)
    
    return names_train[0], names_test[0]



def fit_models(X_train, y_train, X_test, y_test, n, base_path):
    # Define SMOTE and save generated set
    min_class_size = y_train.value_counts().min()
    #if min_class_size > 1:
    try:
        sm = SMOTE(random_state=42, k_neighbors=min(min_class_size - 1, 5))
        X_train, y_train = sm.fit_resample(X_train, y_train)
    except:
        print(f'SMOTE was unsuccessful for {n} model')
    
    X_tr = X_train
    X_t = X_test
    y_tr = y_train
    y_t = y_test
    X_train.to_csv(f'{base_path}/{n}/X_train.csv', index=True, sep=';')
    y_train.to_csv(f'{base_path}/{n}/y_train.csv', index=True, sep=';')
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


    # Initialize the model
    model = RandomForestClassifier()

    # Initialize the grid search parameters
    param_grid = {'max_depth': [1, 5, 10, 15],
                  'n_estimators': [40, 50, 100, 150],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4]}

    # Initialize the number of folds for cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    scoring_metrics = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc'
    }

    # Run grid search with multiple scoring metrics
    grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring=scoring_metrics, refit='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best estimator based on the refit metric
    model = grid_search.best_estimator_

    # Get the predicted values for the full cross-validated model
    y_pred_kfold = cross_val_predict(model, X_train, y_train, cv=kfold)

    # Calculate the k-fold metrics
    accuracy_kfold = accuracy_score(y_train, y_pred_kfold)
    precision_kfold = precision_score(y_train, y_pred_kfold, average='macro')
    recall_kfold = recall_score(y_train, y_pred_kfold, average='macro')
    f1_kfold = f1_score(y_train, y_pred_kfold, average='macro')
    confusion_kfold = confusion_matrix(y_train, y_pred_kfold)

    # Evaluate the model with external validation
    y_pred_ext = model.predict(X_test)
    #print(y_pred_ext)
    #print(X_test)

    # Calculate the external validation metrics
    accuracy_ext = accuracy_score(y_test, y_pred_ext)
    precision_ext = precision_score(y_test, y_pred_ext, average='macro')
    recall_ext = recall_score(y_test, y_pred_ext, average='macro')
    f1_ext = f1_score(y_test, y_pred_ext, average='macro')
    confusion_ext = confusion_matrix(y_test, y_pred_ext)

    params = str(grid_search.best_params_)
        # Calculate the ROC-AUC for k-fold cross-validation
    #fpr_kfold, tpr_kfold, _ = roc_curve(y_train, y_pred_kfold)
    #roc_auc_kfold = auc(fpr_kfold, tpr_kfold)
    y_pred_kfold_prob = cross_val_predict(model, X_train, y_train, cv=kfold, method = 'predict_proba')
    roc_auc_kfold = roc_auc_score(y_train, y_pred_kfold_prob, multi_class='ovo', average='macro')

    # Calculate the ROC-AUC for external validation
    y_pred_ext_prob = model.predict_proba(X_test)
    #fpr_ext, tpr_ext, _ = roc_curve(y_test, y_pred_ext)
    #roc_auc_ext = auc(fpr_ext, tpr_ext)
    roc_auc_ext = roc_auc_score(y_test, y_pred_ext_prob, multi_class='ovo', average='macro')
    """
    # Plot the ROC-AUC curve for k-fold cross-validation
    plt.figure(dpi=300)
    plt.plot(fpr_kfold, tpr_kfold, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_kfold:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{base_path}/{n}/ROC-AUC Curve (k-fold)')
    plt.legend(loc="lower right")
    plt.savefig(f'{base_path}/{n}/roc_auc_kfold.png')
    plt.close()

    # Plot the ROC-AUC curve for external validation
    plt.figure(dpi=300)
    plt.plot(fpr_ext, tpr_ext, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_ext:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{base_path}/{n}/ROC-AUC Curve (External)')
    plt.legend(loc="lower right")
    plt.savefig(f'{base_path}/{n}/roc_auc_ext.png')
    plt.close()
    """
    # Save the model to a file
    filename = f"{base_path}/{n}/model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    # Save confusion matrices as images
    fig, ax = plt.subplots(dpi=600)
    disp_kfold = ConfusionMatrixDisplay(confusion_kfold)
    disp_kfold.plot(ax=ax)
    plt.savefig(f'{base_path}/{n}/confusion_matrix_kfold.png')
    plt.close(fig)

    fig, ax = plt.subplots(dpi=600)
    disp_ext = ConfusionMatrixDisplay(confusion_ext)
    disp_ext.plot(ax=ax)
    plt.savefig(f'{base_path}/{n}/confusion_matrix_ext.png')
    plt.close(fig) 
    tr_dl, test_dl = domain_applcability(X_tr, X_t, y_tr, y_t)

    # Create a dataframe with the results
    result_model = pd.DataFrame({'Y_column': [n],
                                 'Model': ['RF'],
                                 'Hyperparameters': [params],
                                 'Accuracy_kfold': [accuracy_kfold],
                                 'Precision_kfold': [precision_kfold],
                                 'Recall_kfold': [recall_kfold],
                                 'F1_kfold': [f1_kfold],
                                 'ROC_AUC_kfold': [roc_auc_kfold],
                                 'Confusion Matrix (k-fold)': [str(confusion_kfold)],
                                 'Accuracy_ext': [accuracy_ext],
                                 'Precision_ext': [precision_ext],
                                 'Recall_ext': [recall_ext],
                                 'F1_ext': [f1_ext],
                                 'ROC_AUC_ext': [roc_auc_ext],
                                 'Confusion Matrix (External)': [str(confusion_ext)],
                                 'NOT in AD train': [str(tr_dl)],
                                 'NOT in AD test': [str(test_dl)]})
    result_model.set_index('Y_column', inplace=True)
    f_importance = model.feature_importances_
    importance_df = pd.DataFrame(f_importance, columns=['Importance'])

    # Save the DataFrame with Feature Imprtance values to CSV
    importance_df.to_csv(f'{base_path}/{n}/f_importance.csv', index=True, sep=';')


          
    return result_model


#Read the Data

X_all = pd.read_excel('Data.xlsx', sheet_name='X_all', index_col=0)
y_all = pd.read_excel('Data.xlsx', sheet_name='y_all', index_col=0)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
scaler.set_output(transform='pandas')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set your desired path where you want to create the folders
base_path = "RF_TiO2_models"

# Loop through the range of folder names you want to create
for i in range(1, 622):
    folder_name = str(i)
    folder_path = os.path.join(base_path, folder_name)
    
    # Create the folder if it doesn't already exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


#Create dataframe for the results
result_models = pd.DataFrame(columns=['Y_column', 'Model', 'Hyperparameters', 'Accuracy_kfold', 'Precision_kfold', 'Recall_kfold', 'F1_kfold', 'ROC_AUC_kfold', 'Confusion Matrix (k-fold)', 'Accuracy_ext', 'Precision_ext', 'Recall_ext', 'F1_ext', 'ROC_AUC_ext', 'Confusion Matrix (External)', 'NOT in AD train', 'NOT in AD test'])
result_models.set_index('Y_column', inplace=True)

# Configure the logging
logging.basicConfig(level=logging.INFO)
for n in range(1, 622):
    clear_output(wait=True)
    y_train = Y_train[[n]]
    y_test = Y_test[[n]]
    
    #try:
    res = fit_models(X_train_scaled, y_train, X_test_scaled, y_test, n, base_path)
        
        # Append the result of the current model to the result_models DataFrame
    result_models = pd.concat([result_models, res])
    logging.info(f"Processed model {n}")
    """
    except:
        res = pd.DataFrame({'Y_column': [n],
                                 'Model': ['NaN'],
                                 'Hyperparameters': ['NaN'],
                                 'Accuracy_kfold': ['NaN'],
                                 'Precision_kfold': ['NaN'],
                                 'Recall_kfold': ['NaN'],
                                 'F1_kfold': ['NaN'],
                                 'ROC_AUC_kfold': ['NaN'],
                                 'Confusion Matrix (k-fold)': ['NaN'],
                                 'Accuracy_ext': ['NaN'],
                                 'Precision_ext': ['NaN'],
                                 'Recall_ext': ['NaN'],
                                 'F1_ext': ['NaN'],
                                 'ROC_AUC_ext': ['NaN'],
                                 'Confusion Matrix (External)': ['NaN'],
                                 'NOT in AD train': ['NaN'],
                                 'NOT in AD test': ['NaN']})
        res.set_index('Y_column', inplace=True)
        result_models = pd.concat([result_models, res])
        logging.info(f"model {n} wasn`t trained")
    
    """
# Save the results dataframe as a CSV file
result_models.to_csv(f'{base_path}/results.csv', index=True, sep=';')