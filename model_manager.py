import os
import numpy as np
import io
from typing import Dict, Any
from model import ShapeClassifier, MLP

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
        self.model_path = os.path.join('saved_models', 'shape_classifier')
        self.cargar_modelo()
    
    def cargar_modelo(self) -> None:
        """Carga el modelo desde disco si existe"""
        if os.path.exists(self.model_path + '.npz') and os.path.exists(self.model_path + '_classes.json'):
            self.classifier.load_model(self.model_path)
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
        #carga dataset unificado
        X, y, class_names = self._load_dataset_unificado(dataset_path)
        self.classifier.class_names = class_names

        print(f"Clases encontradas: {class_names}")
        print(f"Número de imágenes: {len(X)}")

        if len(X) == 0:
            raise ValueError("Dataset vacío. Guarda imágenes antes de entrenar")
        
        y_one_hot = np.eye(len(class_names))[y]

        if not self.classifier.model or self.classifier.model.layer_sizes[-1] != len(class_names):
            self.classifier.model = MLP([10000, 128, 64, len(class_names)])

        history = self.classifier.model.train(X, y_one_hot, epochs=epochs)
        return {
            'epochs': epochs,
            'final_loss': history['loss'][-1],
            'final_accuracy': history['accuracy'][-1]
        }
    
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
        self.classifier.save_model(self.model_path)
        print(" Modelo guardado correctamente")

    def _load_dataset_unificado(self, dataset_path: str):
        """
        Carga todas las imagenes de todas las clases
        """
        class_names = sorted([
            d for d in os.listdir(dataset_path) 
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
        X, y = [], []

        for idx, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_path, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    with open(img_path, 'rb') as f:
                        img_array = self.classifier.preprocess_image(f.read())
                        X.append(img_array[0])
                        y.append(idx)
        return np.array(X), np.array(y), class_names