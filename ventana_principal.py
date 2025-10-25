import tkinter as tk
from tkinter import ttk
import os
import io
import numpy as np
from PIL import Image, ImageDraw

class VentanaPrincipal:
    """
    Ventana principal para dibujar y clasificar formas.
    Solo se enfoca en la interacción de dibujo y predicción.
    """
    
    def __init__(self, root, model_manager):
        """
        Inicializa la ventana principal
        
        Args:
            root: Ventana principal de tkinter
            model_manager: Instancia de ModelManager para acceder al modelo
        """
        self.root = root
        self.model_manager = model_manager
        
        # Configuración de la ventana
        self.root.title("Perceptrón - Clasificador de Formas")
        self.root.geometry("800x600")
        
        # Variables de dibujo
        self.canvas_size = (400, 400)
        self.last_x = None
        self.last_y = None
        self.drawing = False
        
        # Configurar interfaz
        self.setup_ui()
        
    def setup_ui(self) -> None:
        """Configura todos los elementos de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas de dibujo
        self.setup_canvas(main_frame)
        
        # Controles
        self.setup_controles(main_frame)
        
    
    def setup_canvas(self, parent) -> None:
        """Configura el área de dibujo"""
        canvas_frame = ttk.LabelFrame(parent, text="Zona de Dibujo", padding=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size[0],
            height=self.canvas_size[1],
            bg='black',
            highlightthickness=1,
            highlightbackground='white'
        )
        self.canvas.pack(pady=10)
        
        # Eventos de dibujo 
        self.canvas.bind("<Button-1>", self.comenzar_a_dibujar)        
        self.canvas.bind("<B1-Motion>", self.dibujar)
        self.canvas.bind("<ButtonRelease-1>", self.parar_de_dibujar)  
    
        texto_abajo = ttk.Label(
        canvas_frame, 
        
        text="""Este perceptron solo reconoce números del 0 al 9 y figuras como:
        Circulo, Triangulo, Cuadrado, Corazón, Estrella """,
        font=("Arial", 9),
        foreground="red",
        
    )
        texto_abajo.pack(side=tk.TOP, pady=5)
    def setup_controles(self, parent) -> None:
        """Configura los botones y área de resultados"""
        controls_frame = ttk.LabelFrame(parent, text="Controles", padding=10)
        controls_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        # Botones de acción
        ttk.Button(controls_frame, text="Limpiar Canvas", 
                  command=self.limpiar_canvas).pack(pady=5, fill=tk.X)
        
        ttk.Button(controls_frame, text="Clasificar Forma", 
                  command=self.clasificar_imagen).pack(pady=5, fill=tk.X)
        
        ttk.Button(controls_frame, text="Ir a Entrenamiento", 
                  command=self.abrir_entrenamiento).pack(pady=5, fill=tk.X)
        
        # Área de resultados
        ttk.Label(controls_frame, text="Resultados:").pack(anchor=tk.W, pady=(10, 0))
        self.result_text = tk.Text(controls_frame, height=15, width=35)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def abrir_entrenamiento(self) -> None:
        """Abre la ventana de entrenamiento"""
        from ventana_entrenamiento import VentanaEntrenamiento
        
        self.root.withdraw()  # Ocultar ventana actual
        ventana_entreno = tk.Toplevel()
        VentanaEntrenamiento(ventana_entreno, self.root, self.model_manager)
    
    # metodos de dibujo
    def comenzar_a_dibujar(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def dibujar(self, event):
        if self.drawing and self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=8, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE
            )
            self.last_x = event.x
            self.last_y = event.y
    
    def parar_de_dibujar(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def limpiar_canvas(self):  # 
        self.canvas.delete("all")
        self.log("Canvas limpiado")
        self.result_text.delete(1.0, tk.END) #limpiar resultados    
    
    def get_canvas_image(self):
        """Convertir canvas a imagen PIL"""
        # Crear imagen PIL en blanco y negro
        img = Image.new('L', self.canvas_size, color=0)
        draw = ImageDraw.Draw(img)
        
        # Obtener todas las líneas del canvas y dibujarlas en la imagen
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:  # Línea
                # Convertir coordenadas a enteros
                int_coords = [int(coord) for coord in coords]
                draw.line(int_coords, fill=255, width=8)
        
        return img
    
    def guardar_imagen(self, label): 
        """Guardar imagen en el dataset"""
        try:
            img = self.get_canvas_image()
            
            # Crear carpeta si no existe
            dataset_folder = 'dataset'
            label_folder = os.path.join(dataset_folder, label)
            os.makedirs(label_folder, exist_ok=True)
            
            # Guardar imagen
            timestamp = np.random.randint(10000, 99999)  # Nombre único simple
            filename = f"{label}_{timestamp}.png"
            filepath = os.path.join(label_folder, filename)
            
            img.save(filepath)
            self.log(f" Imagen guardada como: {filename}")
            self.limpiar_canvas()
            
        except Exception as e:
            self.log(f" Error guardando imagen: {str(e)}")
    
    def clasificar_imagen(self):
        """Clasificar la imagen dibujada"""
        try:
            #verificacion
            if not self.model_manager.esta_listo():
                self.log(" Modelo no cargado. Entrena el modelo primero.")
                return
            
            img = self.get_canvas_image()
            
            # Convertir a bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            # predecir
            prediction = self.model_manager.predecir_imagen(img_bytes)
            
            # Mostrar resultados
            result = f" Resultado: {prediction['class']}\n"
            result += f" Confianza: {prediction['confidence']*100:.1f}%\n\n"
            result += "Probabilidades:\n"
            
            for cls, prob in prediction['probabilities'].items():
                result += f"  • {cls}: {prob*100:.1f}%\n"
            
            self.log(result)
            
        except Exception as e:
            self.log(f" Error en clasificación: {str(e)}")
    
    def log(self, mensaje: str) -> None:
        """Agrega un mensaje al área de resultados"""
        self.result_text.insert(tk.END, f"{mensaje}\n")
        self.result_text.see(tk.END)