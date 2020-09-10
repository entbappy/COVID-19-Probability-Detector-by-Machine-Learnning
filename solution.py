import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# Train Test Splitting
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]

if __name__ == "__main__":
    data1 = pd.read_csv('corona.csv')
    train, test = data_split(data1, 0.2)
    
    #Features
    X_train = train[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()
    X_test = test[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()

    #Labels
    Y_train = train[['probability']].to_numpy().reshape(1685,)
    Y_test = test[['probability']].to_numpy().reshape(421,)
    
    # Now Train the data
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)

    file = open('model.pkl','wb')

    pickle.dump(clf, file)

    file.close()
    
    

    
