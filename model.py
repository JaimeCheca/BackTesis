import torch
import torch.nn as nn
import torch.nn.functional as F

class FirePredictionModel(nn.Module):
    def __init__(self):
        super(FirePredictionModel, self).__init__()

        # Procesamiento de Datos Meteorológicos (Entrada 1)
        # Input shape: (batch_size, channels=6, depth=6, height=23, width=19)
        self.meteo_conv3d_1 = nn.Conv3d(in_channels=6, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.meteo_pool3d_1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.meteo_conv3d_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.meteo_pool3d_2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Procesamiento de Datos de Incendios (Entrada 2)
        # Input shape: (batch_size, channels=1, height=23, width=19)
        self.fire_conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.fire_pool2d_1 = nn.MaxPool2d(kernel_size=2)
        self.fire_conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fire_pool2d_2 = nn.MaxPool2d(kernel_size=2)

        # Calcular el tamaño después de las capas convolucionales y de pooling
        # Para meteo data
        # Input: (batch_size, 6, 6, 23, 19)
        # After conv3d_1 and pool3d_1: (batch_size, 32, 6, 11, 9)
        # After conv3d_2 and pool3d_2: (batch_size, 64, 6, 5, 4)
        self.flatten_meteo_size = 64 * 6 * 5 * 4  # 64 canales, 6 profundidad, 5 altura, 4 ancho

        # Para fire data
        # Input: (batch_size, 1, 23, 19)
        # After conv2d_1 and pool2d_1: (batch_size, 32, 11, 9)
        # After conv2d_2 and pool2d_2: (batch_size, 64, 5, 4)
        self.flatten_fire_size = 64 * 5 * 4  # 64 canales, 5 altura, 4 ancho

        # Capas Densas después de la Concatenación
        self.fc1 = nn.Linear(self.flatten_meteo_size + self.flatten_fire_size, 256)
        self.dropout = nn.Dropout(p=0.5)  # Dropout con probabilidad del 50%
        self.fc2 = nn.Linear(256, 23 * 19)  # Salida para cada píxel

    def forward(self, x_meteo, x_fire):
        # Procesamiento de Datos Meteorológicos
        x_meteo = F.relu(self.meteo_conv3d_1(x_meteo))  # (batch_size, 32, 6, 23, 19)
        x_meteo = self.meteo_pool3d_1(x_meteo)           # (batch_size, 32, 6, 11, 9)
        x_meteo = F.relu(self.meteo_conv3d_2(x_meteo))  # (batch_size, 64, 6, 11, 9)
        x_meteo = self.meteo_pool3d_2(x_meteo)           # (batch_size, 64, 6, 5, 4)
        x_meteo = x_meteo.view(x_meteo.size(0), -1)     # (batch_size, 64*6*5*4)

        # Procesamiento de Datos de Incendios
        x_fire = F.relu(self.fire_conv2d_1(x_fire))      # (batch_size, 32, 23, 19)
        x_fire = self.fire_pool2d_1(x_fire)              # (batch_size, 32, 11, 9)
        x_fire = F.relu(self.fire_conv2d_2(x_fire))      # (batch_size, 64, 11, 9)
        x_fire = self.fire_pool2d_2(x_fire)              # (batch_size, 64, 5, 4)
        x_fire = x_fire.view(x_fire.size(0), -1)        # (batch_size, 64*5*4)

        # Concatenación
        x = torch.cat((x_meteo, x_fire), dim=1)          # (batch_size, 64*6*5*4 + 64*5*4)

        # Capas Densas
        x = F.relu(self.fc1(x))                          # (batch_size, 256)
        x = self.dropout(x)  
        x = self.fc2(x)                                  # (batch_size, 23*19)
        x = torch.sigmoid(x)                             # (batch_size, 23*19)
        x = x.view(-1, 1, 23, 19)                        # (batch_size, 1, 23, 19)

        return x