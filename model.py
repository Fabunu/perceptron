import numpy as np
import os
import json
import io
import math
from typing import List, Tuple, Dict

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Pillow no está instalado")

class MLP:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Inicializar pesos con He initialization
        for i in range(len(layer_sizes)-1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            self.weights.append(np.random.randn(input_size, output_size) * np.sqrt(2./input_size))
            self.biases.append(np.zeros((1, output_size)))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray):
        layer_activations = [X]
        
        # Forward pass through hidden layers
        for i in range(len(self.weights)-1):
            X = self.relu(np.dot(X, self.weights[i]) + self.biases[i])
            layer_activations.append(X)
        
        # Output layer (no activation yet)
        logits = np.dot(X, self.weights[-1]) + self.biases[-1]
        return self.softmax(logits), layer_activations
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01) -> Dict:
        history = {'loss': [], 'accuracy': []}
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            epoch_loss, correct = 0, 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred, layer_activations = self.forward(X_batch)
                
                # Calculate loss and accuracy
                loss = -np.mean(np.sum(y_batch * np.log(y_pred + 1e-15), axis=1))
                epoch_loss += loss * X_batch.shape[0]
                correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                
                # Backward pass
                error = (y_pred - y_batch) / X_batch.shape[0]
                self._backward(X_batch, error, layer_activations, learning_rate)
            
            # Record history
            history['loss'].append(epoch_loss / n_samples)
            history['accuracy'].append(correct / n_samples)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Loss={history['loss'][-1]:.4f}, Accuracy={history['accuracy'][-1]:.4f}")
        
        return history
    
    def _backward(self, X: np.ndarray, error: np.ndarray, layer_activations: List[np.ndarray], learning_rate: float) -> None:
        # Output layer
        delta = error
        self.weights[-1] -= learning_rate * np.dot(layer_activations[-1].T, delta)
        self.biases[-1] -= learning_rate * np.sum(delta, axis=0, keepdims=True)
        
        # Hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * (layer_activations[i+1] > 0)
            self.weights[i] -= learning_rate * np.dot(layer_activations[i].T, delta)
            self.biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)
    
    def save_model(self, filepath: str) -> None:
        npz_data = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            npz_data[f"W{i}"] = w
            npz_data[f"b{i}"] = b
        np.savez(filepath, **npz_data)
    
    @classmethod
    def load_model(cls, filepath: str):
        with np.load(f"{filepath}.npz") as data:
            weights, biases = [], []
            i = 0
            while f"W{i}" in data and f"b{i}" in data:
                weights.append(data[f"W{i}"])
                biases.append(data[f"b{i}"])
                i += 1
            
            layer_sizes = [weights[0].shape[0]]
            for w in weights:
                layer_sizes.append(w.shape[1])
            
            mlp = cls(layer_sizes)
            mlp.weights = weights
            mlp.biases = biases
            return mlp

class ShapeClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.class_names = []
        self.img_size = (100, 100)
        if model_path:
            self.load_model(model_path)
    
    def preprocess_image(self, img_data: bytes) -> np.ndarray:
        if not PIL_AVAILABLE:
            raise ImportError("Pillow no está instalado")
            
        img = Image.open(io.BytesIO(img_data)).convert("L")
        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        
        # Normalización
        mean = np.mean(img_array)
        std = np.std(img_array)
        if std > 0:
            img_array = (img_array - mean) / std
        else:
            img_array = img_array - mean
        
        return img_array.flatten().reshape(1, -1)
    
    def predict_image(self, img_data: bytes) -> Dict:
        if not self.model:
            raise ValueError("Model not loaded")
        
        img_array = self.preprocess_image(img_data)
        y_pred, _ = self.model.forward(img_array)
        probabilities = y_pred[0]
        class_idx = np.argmax(probabilities)
        
        if class_idx >= len(self.class_names):
            raise ValueError("Predicted class index out of bounds")
        
        return {
            'class': self.class_names[class_idx],
            'confidence': float(probabilities[class_idx]),
            'probabilities': {name: float(p) for name, p in zip(self.class_names, probabilities)}
        }
    
    def train_model(self, dataset_path: str, epochs: int = 50) -> Dict:
        X, y, class_names = self._load_dataset(dataset_path)
        self.class_names = class_names
        y_onehot = np.eye(len(class_names))[y]
        
        # Arquitectura de la red
        self.model = MLP([10000, 128, 64, len(class_names)])
        history = self.model.train(X, y_onehot, epochs=epochs)
        
        return {
            'epochs': epochs,
            'final_loss': history['loss'][-1],
            'final_accuracy': history['accuracy'][-1]
        }
    
    def _load_dataset(self, dataset_path: str):
        class_names = sorted([d for d in os.listdir(dataset_path) 
                            if os.path.isdir(os.path.join(dataset_path, d))])
        X, y = [], []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_path, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    with open(img_path, 'rb') as f:
                        img_array = self.preprocess_image(f.read())
                        X.append(img_array[0])
                        y.append(class_idx)
        
        return np.array(X), np.array(y), class_names
    
    def save_model(self, model_path: str) -> None:
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        
        # Guardar metadatos
        with open(f"{model_path}_classes.json", 'w') as f:
            json.dump({'class_names': self.class_names, 'img_size': self.img_size}, f)
        
        # Guardar modelo
        self.model.save_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        # Cargar metadatos
        with open(f"{model_path}_classes.json", 'r') as f:
            meta = json.load(f)
            self.class_names = meta['class_names']
            self.img_size = tuple(meta['img_size'])
        
        # Cargar modelo
        self.model = MLP.load_model(model_path)

def crear_imagen_forma(tipo: str, size: Tuple[int, int] = (100, 100)) -> bytes:
    """
    Crea una imagen PNG en bytes con diferentes formas blancas sobre fondo negro.
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow no está instalado")
        
    img = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(img)
    w, h = size
    margin = 20

    if tipo == 'cuadrado':
        draw.rectangle([margin, margin, w - margin, h - margin], fill=255)
    elif tipo == 'triangulo':
        draw.polygon([(w // 2, margin), (margin, h - margin), (w - margin, h - margin)], fill=255)
    elif tipo == 'circulo':
        draw.ellipse([margin, margin, w - margin, h - margin], fill=255)
    elif tipo == 'estrella':
        # Estrella de 5 puntas
        center_x, center_y = w // 2, h // 2
        outer_radius = min(w, h) // 2 - margin
        inner_radius = outer_radius // 2
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * 2 * math.pi / 10
            radius = inner_radius if i % 2 == 0 else outer_radius
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=255)
    elif tipo == 'corazon':
        # Forma de corazón simplificada
        points = [
            (w // 2, margin),  # Top center
            (w - margin, h // 3),  # Right middle
            (w // 2, h - margin),  # Bottom center
            (margin, h // 3)  # Left middle
        ]
        draw.polygon(points, fill=255)
    else:
        raise ValueError("Tipo debe ser 'cuadrado', 'triangulo', 'circulo', 'estrella' o 'corazon'")

    with io.BytesIO() as output:
        img.save(output, format='PNG')
        return output.getvalue()
    
def crear_imagen_numero(numero: str, size: Tuple[int, int] = (100, 100)) -> bytes:
    """
    Crea una imagen PNG en bytes con numeros
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow no está instalado")
    
    img = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(img)
    w, h = size

    #calcular posicion para centrar
    bbox = draw.textbbox((0, 0), numero)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (w - text_width) // 2
    y = (h - text_height) // 2

    #dibujar numero
    draw.text((x, y), numero, fill=255)
    with io.BytesIO() as output:
        img.save(output, format='PNG')
        return output.getvalue()

if __name__ == "__main__":
    if PIL_AVAILABLE:
        # Crear dataset de ejemplo si no existe
        formas = ['cuadrado', 'triangulo', 'circulo', 'estrella', 'corazon']
        numeros = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        dataset_path = 'dataset'
        
        for forma in formas:
            forma_folder = os.path.join(dataset_path, forma)
            os.makedirs(forma_folder, exist_ok=True)
            
            # Crear algunas imágenes de ejemplo
            for i in range(5):
                img_bytes = crear_imagen_forma(forma)
                with open(os.path.join(forma_folder, f'{forma}_{i}.png'), 'wb') as f:
                    f.write(img_bytes)
        
        #imagenes de numeros
        for numero in numeros:
            numero_folder = os.path.join(dataset_path, numero)
            os.makedirs(numero_folder, exist_ok=True)
            
            for i in range(5):
                img_bytes = crear_imagen_numero(numero)
                with open(os.path.join(numero_folder, f'numero_{numero}_{i}.png'), 'wb') as f:
                    f.write(img_bytes)
                    
        print("Dataset de ejemplo creado!")
    else:
        print("Instala Pillow primero: pip install pillow")