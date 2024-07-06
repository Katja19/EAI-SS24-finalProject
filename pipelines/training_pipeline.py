from zenml import pipeline
from steps import load_data, create_model, evaluate, update_data,split_data,feature_engineering

@pipeline(enable_cache=False)
def training_pipeline():
    
    print("Updating data and loading...")
    # Daten aktualisieren und DANACH laden
    load_data.after(update_data)
    update_data()
    
    # Dataensatz zum Trainieren des Modells laden
    dataset = load_data()
    
    print('Splitting data...')
    # Daten in Trainings- und Testdaten aufteilen
    X_train,X_test,y_train,y_test = split_data(dataset,"pedestrians_count")
    
    print('Feature-Engineering...')
    # Feature-Engineering der X-Daten
    X_train,X_test = feature_engineering(X_train,X_test)
    
    # pirnt the first 5 rows of the data
    # print(X_train.head())
    
    print('Creating model...')
    # Modell erstellen und trainieren
    model,in_sample_mae = create_model(X_train,y_train)
    mae = evaluate(model=model,X_test=X_test,y_test=y_test)