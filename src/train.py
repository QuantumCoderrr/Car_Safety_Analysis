from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def split_data(data, target_col="class", test_size=0.3, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors (KNN) classifier.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train a Random Forest classifier.
    """
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)
    return rf
