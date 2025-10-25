import os
import numpy as np
import io
from typing import Dict, Any

# Importar tus clases del modelo
try:
    from model import ShapeClassifier
except ImportError:
    # Si tu modelo está en el mismo archivo, necesitarás ajustar esto
    print("Error: No se pudo importar ShapeClassifier")

class ModelManager:
    """
    Gestiona el modelo de clasificación de formas.
    Proporciona una interfaz simple para entrenar y predecir.
    """
    
    def __init__(self):
        self.classifier = ShapeClassifier()
        self.cargar_modelo()
    
    def cargar_modelo(self) -> None:
        """Carga el modelo desde disco si existe"""
        model_path = os.path.join('saved_models', 'shape_classifier')
        if os.path.exists(model_path + '.npz') and os.path.exists(model_path + '_classes.json'):
            self.classifier.load_model(model_path)
            print(" Modelo cargado correctamente desde disco")
        else:
            print(" No se encontró modelo preentrenado")
    
    def entrenar_modelo(self, epochs: int = 50) -> Dict[str, Any]:
        """
        Entrena el modelo con las imágenes del dataset
        
        Args:
            epochs: Número de épocas de entrenamiento
            
        Returns:
            Dict con resultados del entrenamiento
        """
        dataset_path = 'dataset'
        return self.classifier.train_model(dataset_path=dataset_path, epochs=epochs)
    
    def predecir_imagen(self, imagen_bytes: bytes) -> Dict[str, Any]:
        """
        Clasifica una imagen
        
        Args:
            imagen_bytes: Imagen en formato bytes
            
        Returns:
            Dict con resultados de la predicción
        """
        return self.classifier.predict_image(imagen_bytes)
    
    def esta_listo(self) -> bool:
        """Verifica si el modelo está cargado y listo"""
        return hasattr(self.classifier, 'model') and self.classifier.model is not None
    
    def guardar_modelo(self) -> None:
        """Guarda el modelo en disco"""
        model_path = os.path.join('saved_models', 'shape_classifier')
        self.classifier.save_model(model_path)
        print(" Modelo guardado correctamente")