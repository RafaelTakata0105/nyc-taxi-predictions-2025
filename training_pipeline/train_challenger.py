import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pathlib
import pandas as pd
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from prefect import flow, task

@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name="Hyperparameter Tunning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv, model_obj):
    
    mlflow.sklearn.autolog()
    
    # 1. Determinar el tipo de modelo y configurar la búsqueda (Study)
    if isinstance(model_obj, RandomForestRegressor):
        model_name = "Random Forest Regressor"
        
        def objective(trial: optuna.trial.Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0, log=False),
                "random_state": 42,
                "n_jobs": -1,
            }

            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_family", "random_forest_regressor")
                mlflow.log_params(params)
                
                model = model_obj.__class__(**params) # Usar __class__ para crear nueva instancia
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val) # CORREGIDO: Usar X_val para predecir
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("rmse", rmse)
                
                signature = infer_signature(X_val, y_pred)
                mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5], signature=signature)
            return rmse

    elif isinstance(model_obj, GradientBoostingRegressor):
        model_name = "Gradient Boosting Regressor"

        def objective(trial: optuna.trial.Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', None]),
                "random_state": 42,
            }

            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_family", "gradient_boosting_regressor")
                mlflow.log_params(params)
                
                model = model_obj.__class__(**params) # Usar __class__ para crear nueva instancia
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val) # CORREGIDO: Usar X_val para predecir
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("rmse", rmse)
                
                signature = infer_signature(X_val, y_pred)
                mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5], signature=signature)
            return rmse

    else:
        # Lanza un error si se pasa un modelo no soportado
        raise ValueError(f"Modelo no soportado para Optuna: {type(model_obj)}")

    # 2. Ejecutar Optuna (Esta sección ahora siempre se alcanza después de definir 'objective')
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler) # Ahora 'study' está definida
    
    with mlflow.start_run(run_name=f"{model_name} Hyperparameter Optimization (Optuna)", nested=True):
        study.optimize(objective, n_trials=3)

    # 3. Recuperar y retornar los mejores hiperparámetros
    best_params = study.best_params
    
    # Manejar campos fijos para los modelos de sklearn
    if "max_depth" in best_params:
        best_params["max_depth"] = int(best_params["max_depth"])
    
    # El parámetro 'objective' y 'seed' no es necesario en Scikit-learn como lo era en XGBoost.
    # Simplemente mantenemos la semilla si el modelo la soporta.
    best_params["random_state"] = 42

    return best_params

@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params, model_obj) -> None:
    """train a model with best hyperparams and write everything out"""

    if isinstance(model_obj, GradientBoostingRegressor):
        model_family = "Gradient Boosting Regressor"
    elif isinstance(model_obj, RandomForestRegressor):
        model_family = "Random Forest Regressor"
    else:
        raise ValueError("Modelo no reconocido.")

    with mlflow.start_run(run_name=f"{model_family} Final Model"):
        mlflow.log_params(best_params)

        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": model_family,
            "feature_set_version": 1,
        })

        # Entrenar el modelo FINAL
        model = model_obj.__class__(**best_params) # Usar __class__
        model.fit(X_train, y_train)

        # Evaluar y registrar la métrica final
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Guardar artefactos
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # Preparar y guardar la signature y el modelo
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature,
        )
    return None

@task(name= "Model Registry")
def save_best_metric_model(experiment_name:str)-> None:
    #Get Run uri of best result:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.rmse ASC"],
        output_format="list") 
    
    if len(runs) > 0:
        best_run = runs[0]
        model_uri_full = f"runs:/{best_run.info.run_id}/model"
        result = mlflow.register_model(
            model_uri=model_uri_full,
            name="workspace.default.nyc-taxi-model-prefect")
    return None

@task(name="Manage Model Aliases")
def manage_model_alias(model_name: str) -> None:
    """Compara todas las versiones registradas del modelo y asigna los alias Champion/Challenger."""
    client = MlflowClient()
    
    # 1. Obtener todas las versiones del modelo registrado en Unity Catalog
    try:
        # Nota: Unity Catalog usa el nombre del modelo como un namespace
        all_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"ERROR: No se pudo encontrar el modelo '{model_name}' en el registro. Ejecute el flujo al menos dos veces. Error: {e}")
        return

    scored_versions = []
    
    # 2. Recorrer las versiones, obtener el RMSE del run asociado
    for version in all_versions:
        run_id = version.run_id
        if not run_id:
            continue
            
        try:
            # Obtener el run para extraer la métrica RMSE
            run_data = client.get_run(run_id)
            # Buscamos 'rmse' que es la métrica registrada en el run final
            rmse = run_data.data.metrics.get("rmse") 
            
            if rmse is not None:
                scored_versions.append({
                    "version": version.version,
                    "rmse": rmse,
                    "aliases": version.aliases,
                    "model_family": run_data.data.tags.get("model_family", "N/A"),
                })
        except Exception:
            # Ignoramos versiones que no tienen el run asociado disponible o métrica.
            pass

    if len(scored_versions) < 2:
        print(f"ADVERTENCIA: Solo se encontraron {len(scored_versions)} versiones válidas con métrica 'rmse'. Se requieren al menos 2.")
        return

    # 3. Ordenar por RMSE (ascendente: el más bajo es el mejor)
    scored_versions.sort(key=lambda x: x["rmse"])
    
    champion = scored_versions[0]
    challenger = scored_versions[1]
    
    for version in scored_versions:
        if "Champion" in version["aliases"]:
            client.delete_model_version_alias(name=model_name, alias="Champion")
        if "Challenger" in version["aliases"]:
            client.delete_model_version_alias(name=model_name, alias="Challenger")
    
    client.set_model_version_alias(name=model_name, alias="Champion", version=champion["version"])
    client.set_model_version_alias(name=model_name, alias="Challenger", version=challenger["version"])
    
    print(f"Champion y Challenger asignados exitosamente.")


def main_flow(year: int, month_train: str, month_val: str) -> None:
    """The main training pipeline for competitive model selection."""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True)
    
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # 1. Load y Transformación de Datos
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # --- Ejecutar Modelos Competidores ---
    
    # 2. Entrenamiento de RANDOM FOREST
    rf_model = RandomForestRegressor(random_state=42)
    rf_best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv, rf_model)
    # Entrenar y obtener el run_id del modelo final
    rf_run_id = train_best_model(X_train, X_val, y_train, y_val, dv, rf_best_params, rf_model)
    register_model_version(rf_run_id, MODEL_REGISTRY_NAME) # <-- Registro de Versión RF

    # 3. Entrenamiento de GRADIENT BOOSTING
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv, gb_model)
    # Entrenar y obtener el run_id del modelo final
    gb_run_id = train_best_model(X_train, X_val, y_train, y_val, dv, gb_best_params, gb_model)
    register_model_version(gb_run_id, MODEL_REGISTRY_NAME) # <-- Registro de Versión GB
    
    # 4. Asignar Champion/Challenger (Compara TODAS las versiones existentes)
    manage_model_alias(MODEL_REGISTRY_NAME)

# --- EJECUCIÓN ---
if __name__ == "__main__":
    main_flow(year=2025, month_train="01", month_val="02")