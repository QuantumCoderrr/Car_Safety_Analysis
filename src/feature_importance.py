import pandas as pd

def calculate_feature_importance(model, feature_names):
    """
    Calculate feature importance from a trained model.
    """
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)
    return importance_df
