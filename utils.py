import os
import yaml
import os
import pandas as pd

def find_project_root():
    current_dir = os.getcwd()
    
    # Subir hasta encontrar el directorio raíz del proyecto que contiene 'config/config.yaml'
    while not os.path.exists(os.path.join(current_dir, 'config', 'config.yaml')):
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):  # Límite de directorios (raíz del sistema)
            raise FileNotFoundError("No se pudo encontrar el archivo 'config.yaml' en la jerarquía.")
    
    return current_dir

def get_config_file():
    prj_root = find_project_root()
    config_path = os.path.join(prj_root,'config', 'config.yaml')

    #Abrimos el archivo con la ruta obtenida
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_input_model():
    prj_root = find_project_root()
    config = get_config_file()
    path_data =  config["paths"]["data_input_model"]
    full_path_data = os.path.join(prj_root,path_data)
    df = pd.read_csv(full_path_data, parse_dates=[0, 1])
    return df

def get_input_model_acortado():
    prj_root = find_project_root()
    config = get_config_file()
    path_data =  config["paths"]["data_input_model_acortado"]
    full_path_data = os.path.join(prj_root,path_data)
    df = pd.read_csv(full_path_data, parse_dates=[0, 1])
    return df

def change_path_separator(strng):
    return strng.replace("\\", "/")

def agregar_cero_inicial(n):
    return f"{n:02}"

def get_shape_final():
    prj_root = find_project_root()
    config = get_config_file()
    grid =  config["config"]["grid_shape"]
    return grid