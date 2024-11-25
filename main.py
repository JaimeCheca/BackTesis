from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
from datetime import datetime, timedelta
from model import FirePredictionModel
from utils import change_path_separator, agregar_cero_inicial, get_shape_final
import torch
from PIL import Image

app = FastAPI()

origins = [
    "http://localhost:3000",  # Ajusta según el origen de tu frontend
]

# Configuración de CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permitir todos los orígenes (cambiar en producción)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

p_train = {
    'prj_root': "C:\\Jaime\\Anaconda\\Proyecto_Tesis",
    'path_output_meteorological': "output/grids_meteorologicas",
    'path_output_serfor': "output/grids_output",
    
}

hyperparams = {
    'inputs'     : ['fecha_met_ini','fecha_met_fin','fecha_output_input','fecha_output_output'],
    'var_meteorologicas': ['lai_hv', 'sp', 't2m', 'tp', 'u10', 'v10'],
}

device = torch.device('cpu')
model = FirePredictionModel()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Clase para el esquema de entrada
class PredictionRequest(BaseModel):
    date_time: str  # Fecha y hora en formato ISO 8601 (e.g., "2023-10-15T12:00:00")

def obtener_paths_meteorologicas(fecha_1, fecha_6, HP, P):
    fecha_inicio = fecha_6
    fecha_fin = fecha_1
    
    total_horas = int((fecha_fin - fecha_inicio).total_seconds() // 3600)
    
    lista_fechas = [fecha_inicio + timedelta(hours=i) for i in range(total_horas + 1)]
    
    lista_path = []
    
    for fecha in lista_fechas:
        año = str(fecha.year)
        mes = str(agregar_cero_inicial(fecha.month))
        dia = str(agregar_cero_inicial(fecha.day))
        hora = str(agregar_cero_inicial(fecha.hour))
        file_name = año + '_' + mes + '_' + dia + '_' + hora + '.png.npy'
        for var in HP["var_meteorologicas"]:
            file_path = os.path.join(P["prj_root"], P["path_output_meteorological"],año,var, file_name)
            lista_path.append(change_path_separator(file_path))
    return lista_path

def load_meteo_data(paths):
    lista_imagenes = []
    for path in paths:
        numpy_array = np.load(path)
        lista_imagenes.append(numpy_array)
        
    images = np.stack(lista_imagenes)
    images = images.reshape(6, 6, 23, 19, 1)
    images = np.transpose(images, axes=(1, 2, 3, 0, 4))
    images = images.reshape(6, 23, 19, 6)
    images = np.transpose(images, axes=(3, 0, 1, 2))
    X_meteo = torch.tensor(images, dtype=torch.float32)
    return X_meteo

def load_fire_data(path):
    # Cargar la imagen
    img = Image.open(path)
    
    # Convertir a escala de grises si es necesario
    img = img.convert('L')
    # Convertir a array NumPy
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    X_fire = torch.tensor(img_array, dtype=torch.float32)
    return X_fire

def load_output_data(path):
    # Cargar la imagen
    img = Image.open(path)
    
    # Convertir a escala de grises si es necesario
    img = img.convert('L')
    # Convertir a array NumPy
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    return img_array

def concatenate_data(pred, real):
    shape = get_shape_final()
    print(real)
    x= shape[0]
    y = shape[1]
    grid_shape = (x, y)
    grid = np.zeros(grid_shape)
    for i in range(x):
        for j in range(y):
            #correcto -> 1, incorrecto -> rojo, prediccionRed -> amarillo
            if (real[i][j] == 1) and (pred[i][j] >= 0.5):
                grid[i][j] = 1
            elif (real[i][j] == 1) and (pred[i][j] < 0.5):
                grid[i][j] = 0.5
            elif (real[i][j] == 0) and (pred[i][j] > 0.1):
                grid[i][j] = 0.2
    return grid


# Endpoint para realizar predicciones
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        fecha = datetime.strptime(request.date_time, "%Y-%m-%dT%H:%M")

        fecha_1 = fecha - timedelta(hours=1)
        fecha_6 = fecha - timedelta(hours=6)
        fecha_5 = fecha + timedelta(hours=5)
        
        lista_path_met = obtener_paths_meteorologicas(fecha_1, fecha_6, hyperparams, p_train)
        X_meteo = load_meteo_data(lista_path_met)
        
        path_fire = f"{str(agregar_cero_inicial(fecha_6.year))}-{str(agregar_cero_inicial(fecha_6.month))}-{str(agregar_cero_inicial(fecha_6.day))} {str(agregar_cero_inicial(fecha_6.hour))}_00_00_{str(agregar_cero_inicial(fecha_1.year))}-{str(agregar_cero_inicial(fecha_1.month))}-{str(agregar_cero_inicial(fecha_1.day))} {str(agregar_cero_inicial(fecha_1.hour))}_00_00.png"
        path_output = f"{str(agregar_cero_inicial(fecha.year))}-{str(agregar_cero_inicial(fecha.month))}-{str(agregar_cero_inicial(fecha.day))} {str(agregar_cero_inicial(fecha.hour))}_00_00_{str(agregar_cero_inicial(fecha_5.year))}-{str(agregar_cero_inicial(fecha_5.month))}-{str(agregar_cero_inicial(fecha_5.day))} {str(agregar_cero_inicial(fecha_5.hour))}_00_00.png"
        path_fire_final = "C:/Jaime/Anaconda/Proyecto_Tesis/output/grids_output/"+path_fire
        path_output_final = "C:/Jaime/Anaconda/Proyecto_Tesis/output/grids_output/"+path_output
        X_fire = load_fire_data(path_fire_final)
        
        X_meteo = X_meteo.reshape(1, 6, 6, 23, 19)
        X_fire = X_fire.reshape(1, 1, 23, 19)
        
        output = model(X_meteo, X_fire)
        output_squeezed = output.squeeze()

        numpy_prediction_array = output_squeezed.detach().numpy()

        prediction_list = numpy_prediction_array.tolist()

        real_list = load_output_data(path_output_final)

        data_concatenada = concatenate_data(numpy_prediction_array, real_list)

        print(data_concatenada)

        # Devolver la predicción al frontend
        return {
            "prediction": prediction_list,
            "real_data": real_list.tolist(),
            "data_concatenada": data_concatenada.tolist()
        }

    except Exception as e:
        print("Error durante la predicción:", str(e))
        raise HTTPException(status_code=400, detail=str(e))