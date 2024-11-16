

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve




#----------------------------SETTINGS------------------------------

dataset = 1 # Set to abalone dataset. Set = 2 to test with contraceptive dataset

#----------------------------------------------------------

if (dataset==2):
    contraceptive_method_choice = fetch_ucirepo(id=30)
    X = contraceptive_method_choice.data.features
    y = contraceptive_method_choice.data.targets
    data = pd.concat([X,y], axis=1)
    data.to_csv("contraceptive.csv", index=False)

    corr_matrix = data.corr()
    correlations = corr_matrix['contraceptive_method'].drop(['contraceptive_method'])  
    top_corr = correlations.abs().nlargest(1)

    #GRADIENT BOOST
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Classification Accuracy: {accuracy:.2f}")

    #NEURAL NETWORK
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    dropout = 0.1
    l2_ = 0.001

    model3 = keras.Sequential([
    layers.Dense(256,activation='relu', kernel_regularizer = l2(l2_)),
    layers.Dropout(dropout),
    layers.Dense(128,activation='relu', kernel_regularizer = l2(l2_)),
    layers.Dropout(dropout),
    layers.Dense(32,activation='relu', kernel_regularizer = l2(l2_)),
    layers.Dropout(dropout),
    layers.Dense(4, activation='softmax', kernel_regularizer = l2(l2_))
    ])

    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model3.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    predictions = model3.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_one_hot, axis=1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    roc = roc_auc_score(y_test_one_hot, predictions)
    print("Test accuracy, ",accuracy)
    print("roc, ",roc) 

else:
    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features
    y = abalone.data.targets
    data = pd.concat([X,y], axis=1)
    data.to_csv("abalone.csv", index=False)

    def classify_age(rings):
        if rings >= 0 and rings <= 7:
            return '0'
        elif rings >= 8 and rings <= 10:
            return '1'  
        elif rings >= 11 and rings <= 15:
            return '2'  
        else:
            return '3'  
        
    data['Age_category'] = data['Rings'].apply(classify_age)
    data['Sex'] = data['Sex'].replace({'M': 0, 'F': 1, 'I': 2})
    X['Sex'] = X['Sex'].replace({'M': 0, 'F': 1, 'I': 2})

    corr_matrix = data.corr()
    correlations = corr_matrix['Rings'].drop(['Rings', 'Age_category'])  
    top_corr = correlations.abs().nlargest(1)
    y = data[['Age_category']]

    #DECISION TREE MODEL
    d = [2,3,4,5,6,7,8,9]
    a = []
    m = []
    for i in range(len(d)):
        X_train, X_test, y_train, y_test = train_test_split(X, data['Age_category'], test_size=0.2, random_state=i)

        dt = DecisionTreeClassifier(random_state=42, max_depth=d[i])
        dt.fit(X_train, y_train)
        test_accuracy = accuracy_score(y_test, dt.predict(X_test))
        a.append(test_accuracy)
        m.append(dt)
        print(test_accuracy, 'depth = ', d[i])

    pos = a.index(max(a))
    plt.figure(figsize=(12, 8)) 
    plot_tree(m[pos], filled=True, feature_names=X.columns.tolist(), class_names=data['Age_category'].unique(), rounded=True)
    plt.title(f'Decision Tree (Depth = {d[pos]})')
    plt.savefig('decisiontree.png')
    tree_rules = export_text(m[pos], feature_names=list(X.columns))
    print(tree_rules)
    #plt.show()
    
    #PRUNING
    # results = {
    #     "Standard Decision Tree": [],
    #     "Pre-Pruned Tree": [],
    #     "Post-Pruned Tree": []
    # }

    # for i in range(5):
    #     X_train, X_test, y_train, y_test = train_test_split(X, data['Age_category'], test_size=0.2, random_state=i)
        
    #     # Standard Decision Tree
    #     std_tree = DecisionTreeClassifier(random_state=42)
    #     std_tree.fit(X_train, y_train)
    #     std_accuracy = accuracy_score(y_test, std_tree.predict(X_test))
    #     results["Standard Decision Tree"].append(std_accuracy)

    #     # Pre-Pruned Tree (via GridSearchCV)
    #     param_grid = {
    #         'max_depth': [5, 10, 15, None],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 2, 4]
    #     }
    #     grid_search = GridSearchCV(
    #         estimator=DecisionTreeClassifier(random_state=42),
    #         param_grid=param_grid,
    #         cv=5,
    #         scoring='accuracy',
    #         verbose=0
    #     )
    #     grid_search.fit(X_train, y_train)
    #     pre_pruned_tree = grid_search.best_estimator_
    #     pre_pruned_accuracy = accuracy_score(y_test, pre_pruned_tree.predict(X_test))
    #     results["Pre-Pruned Tree"].append(pre_pruned_accuracy)

    #     # Post-Pruned Tree
    #     unpruned_tree = DecisionTreeClassifier(random_state=42)
    #     unpruned_tree.fit(X_train, y_train)
    #     path = unpruned_tree.cost_complexity_pruning_path(X_train, y_train)
    #     ccp_alphas = path.ccp_alphas
    #     pruned_accuracies = []

    #     for ccp_alpha in ccp_alphas:
    #         pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    #         pruned_tree.fit(X_train, y_train)
    #         pruned_accuracies.append(accuracy_score(y_test, pruned_tree.predict(X_test)))
    #     best_post_pruned_accuracy = max(pruned_accuracies)
    #     results["Post-Pruned Tree"].append(best_post_pruned_accuracy)

    # for method, accuracies in results.items():
    #     avg_accuracy = sum(accuracies) / len(accuracies)
    #     print(f"{method}: Average Accuracy = {avg_accuracy:.4f}")

    #RANDOM FOREST MODEL
    X_train, X_test, y_train, y_test = train_test_split(X, data['Age_category'], test_size=0.2)

    n = [50,100,200,300,400,500]
    for i in n:
        rf = RandomForestClassifier(random_state=42, n_estimators=i)
        rf.fit(X_train, y_train)

        test_accuracy = accuracy_score(y_test, rf.predict(X_test))
        print("Accuracy",test_accuracy,"\tNumber of trees, ",i)

    #XGBOOST
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train_encoded)
    y_pred = xgb_clf.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"XGBoost Classification Accuracy: {accuracy:.2f}")


    #GRADIENT BOOSTING
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Classification Accuracy: {accuracy:.2f}")

    #NEURAL NETWORK (BASE MODEL)
    X_train, X_test, y_train, y_test = train_test_split(X, data['Age_category'], test_size=0.2, random_state=42)
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    y_test_one_hot

    model1 = keras.Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(4, activation='softmax')
    ])

    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model1.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    predictions = model1.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_one_hot, axis=1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    roc = roc_auc_score(y_test_one_hot, predictions)
    print("Test accuracy, ",accuracy)
    print("roc, ",roc)

    #NEURAL NETWORK (WITH DROPOUTS)

    dropout = 0.1

    model2 = keras.Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(128,activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(32,activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(4, activation='softmax')
    ])

    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    predictions = model2.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_one_hot, axis=1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    roc = roc_auc_score(y_test_one_hot, predictions)
    print("Test accuracy, ",accuracy)
    print("roc, ",roc)

    #NEURAL NETWORK (WITH DROPOUTS AND L2 REGULARIZATION)

    dropout = 0.1
    l2_ = 0.001

    model3 = keras.Sequential([
    layers.Dense(256,activation='relu', kernel_regularizer = l2(l2_)),
    layers.Dropout(dropout),
    layers.Dense(128,activation='relu', kernel_regularizer = l2(l2_)),
    layers.Dropout(dropout),
    layers.Dense(32,activation='relu', kernel_regularizer = l2(l2_)),
    layers.Dropout(dropout),
    layers.Dense(4, activation='softmax', kernel_regularizer = l2(l2_))
    ])

    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model3.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    predictions = model3.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_one_hot, axis=1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    roc = roc_auc_score(y_test_one_hot, predictions)
    print("Test accuracy, ",accuracy)
    print("roc, ",roc)



#Plotting Correlation Matrix
corr_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1)
plt.title('Heatmap of Best Correlation Feature with Target')
plt.savefig('correlation_heatmap.png')
#plt.show()

#Plotting Scatterplot of Best Feature with Target
sns.scatterplot(data=data, x=y.columns[0], y=top_corr.keys()[0], c='blue', alpha=0.3)
plt.savefig('scatterplot.png')
#plt.show()

target = data[str(y.columns[0])]

#Plotting Distribution of Target Variables
sns.histplot(target)
plt.title('Histogram of Target')
plt.savefig('histogram.png')
#plt.show()


