import os
from src.preprocess import load_data, encode_data
from src.train import split_data, train_knn, train_random_forest
from src.evaluate import evaluate_model
from src.feature_importance import calculate_feature_importance
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Paths
    data_path = os.path.join("data", "Car_Safety_Data.csv")
    
    # Load and preprocess the data
    data = load_data(data_path)
    data, label_encoders = encode_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Train KNN model
    knn = train_knn(X_train, y_train)
    knn_accuracy, knn_report, knn_conf_matrix = evaluate_model(knn, X_test, y_test)

    print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
    print("\nKNN Classification Report:\n", knn_report)

    # Train Random Forest model
    rf = train_random_forest(X_train, y_train)
    rf_accuracy, rf_report, rf_conf_matrix = evaluate_model(rf, X_test, y_test)

    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    print("\nRandom Forest Classification Report:\n", rf_report)

    # Feature importance
    importance_df = calculate_feature_importance(rf, X_train.columns)
    print("\nFeature Importances:\n", importance_df)

    # Visualization
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance Analysis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
