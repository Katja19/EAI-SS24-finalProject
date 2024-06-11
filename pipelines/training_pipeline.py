from zenml import pipeline
from steps import load_data, create_model, evaluate, update_data,split_data,feature_engineering
@pipeline(enable_cache=False)
def training_pipeline():
    
    load_data.after(update_data)
    update_data()
    dataset = load_data()
    X_train,X_test,y_train,y_test = split_data(dataset,"pedestrians_count")
    X_train,X_test = feature_engineering(X_train,X_test)
    model,in_sample_mae = create_model(X_train,y_train)
    mae = evaluate(model=model,X_test=X_test,y_test=y_test)