import tkinter as tk
import os
from ventana_principal import VentanaPrincipal
from model_manager import ModelManager


def main():
    # Crear carpetas necesarias
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    # Crear ventana principal
    root = tk.Tk()
    
    # Crear gestor del modelo
    model_manager = ModelManager()
    
    # Crear ventana principal pasando el gestor del modelo
    app = VentanaPrincipal(root, model_manager)
    
    # Iniciar aplicación
    root.mainloop()

if __name__ == "__main__":
    main()