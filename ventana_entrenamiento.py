import tkinter as tk
from tkinter import ttk
import os
import time
import io
import numpy as np
from PIL import Image, ImageDraw

class VentanaEntrenamiento:
    """
    Ventana dedicada al entrenamiento del modelo y gestión del dataset.
    """
    
    def __init__(self, root, ventana_principal, model_manager):
        self.root = root
        self.ventana_principal = ventana_principal
        self.model_manager = model_manager
        
        # Configuración de la ventana
        self.root.title("Entrenamiento del Modelo")
        self.root.geometry("900x700")  # Más ancha para dos columnas
        self.root.protocol("WM_DELETE_WINDOW", self.volver_principal)
        
        # Variables de dibujo
        self.canvas_size = (300, 300)
        self.last_x = None
        self.last_y = None
        self.drawing = False
        
        # Configurar interfaz
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Configura la interfaz de entrenamiento"""
        # Frame principal con dos columnas
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # COLUMNA IZQUIERDA: Canvas para dibujar
        left_frame = ttk.LabelFrame(main_frame, text="Dibujar Formas", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.setup_canvas_entrenamiento(left_frame)
        texto_entrenamiento = ttk.Label(
            left_frame,  
            text="""Instrucciones:
1. Dibuja una figura/número en el recuadro negro
2. Guarda la forma con los botones al costado
   (la forma dibujada tiene que coincidir con el botón)
3. Oprime 'INICIAR ENTRENAMIENTO' para que el programa aprenda
4. Vuelve a la ventana principal y prueba si adivina""",
            font=("Arial", 9),
            foreground="darkgreen",
            background="lightgreen",
            padding=8,
            justify=tk.LEFT
        )
        texto_entrenamiento.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        # COLUMNA DERECHA: Controles de entrenamiento
        right_frame = ttk.LabelFrame(main_frame, text="Controles de Entrenamiento", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controles_entrenamiento(right_frame)
    
    def setup_canvas_entrenamiento(self, parent):
        """Canvas para dibujar en la ventana de entrenamiento"""
        # Canvas de dibujo
        self.canvas = tk.Canvas(
            parent,
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
        
        # Botones de acción del canvas
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Limpiar Canvas", 
                  command=self.limpiar_canvas).pack(side=tk.LEFT, padx=5)
    
    def setup_controles_entrenamiento(self, parent):
        """Configura TODOS los controles de entrenamiento en la columna derecha"""
        
        # 1. CONFIGURACIÓN DE ENTRENAMIENTO
        config_frame = ttk.LabelFrame(parent, text="Configuración", padding=10)
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Épocas de entrenamiento:").grid(row=0, column=0, padx=5, sticky="w")
        self.epochs_entry = ttk.Entry(config_frame, width=10)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.grid(row=0, column=1, padx=5)
        
        # 2. BOTONES PARA GUARDAR FORMAS DIBUJADAS
        formas_frame = ttk.LabelFrame(parent, text="Guardar Forma Dibujada", padding=10)
        formas_frame.pack(fill=tk.X, pady=10)
        
        
        formas_manual = [
            ('Cuadrado', 'cuadrado', '#ff9f1c'),
            ('Triángulo', 'triangulo', '#2ec4b6'), 
            ('Círculo', 'circulo', '#e71d36'),
            ('Estrella', 'estrella', '#ffd166'),
            ('Corazón', 'corazon', '#ff6b6b')
        ]
        
        manual_frame = ttk.Frame(formas_frame)
        manual_frame.pack(fill=tk.X, pady=5)
        
        for text, shape, color in formas_manual:
            btn = tk.Button(
                manual_frame,
                text=text,
                bg=color,
                fg='black',
                command=lambda s=shape: self.guardar_dibujo_actual(s),
                width=10
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        
        # 4. BOTÓN DE ENTRENAMIENTO
        entrenar_frame = ttk.LabelFrame(parent, text="Entrenar Modelo", padding=10)
        entrenar_frame.pack(fill=tk.X, pady=10)
        
        self.btn_entrenar = tk.Button(
            entrenar_frame,
            text="INICIAR ENTRENAMIENTO",
            command=self.iniciar_entrenamiento,
            bg="#38b000",
            fg="white",
            font=("Arial", 12, "bold"),
            width=20,
            height=2
        )
        self.btn_entrenar.pack(pady=10)
        
        # 5. ÁREA DE PROGRESO
        progreso_frame = ttk.LabelFrame(parent, text="Progreso y Logs", padding=10)
        progreso_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.progreso_text = tk.Text(progreso_frame, height=8, width=50)
        scrollbar = ttk.Scrollbar(progreso_frame, command=self.progreso_text.yview)
        self.progreso_text.config(yscrollcommand=scrollbar.set)
        
        self.progreso_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 6. BOTÓN VOLVER
        volver_frame = ttk.Frame(parent)
        volver_frame.pack(pady=10)
        
        ttk.Button(volver_frame, text="Volver a Dibujar", 
                  command=self.volver_principal).pack()
    
    # --- MÉTODOS DE DIBUJO ---
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
    
    def limpiar_canvas(self):
        self.canvas.delete("all")
        self.log("Canvas limpiado")
    
    def get_canvas_image(self):
        """Convierte el canvas a imagen PIL"""
        img = Image.new('L', self.canvas_size, color=0)
        draw = ImageDraw.Draw(img)
        
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:
                int_coords = [int(coord) for coord in coords]
                draw.line(int_coords, fill=255, width=8)
        
        return img
    
    # --- MÉTODOS DE GUARDADO ---
    def guardar_dibujo_actual(self, etiqueta: str):
        """Guarda lo que está dibujado en el canvas"""
        try:
            # Verificar que hay algo dibujado
            if len(self.canvas.find_all()) == 0:
                self.log(" No hay nada dibujado en el canvas")
                return
                
            img = self.get_canvas_image()
            
            # Crear carpeta si no existe
            dataset_folder = 'dataset'
            label_folder = os.path.join(dataset_folder, etiqueta)
            os.makedirs(label_folder, exist_ok=True)
            
            # Guardar imagen
            timestamp = int(time.time() * 1000)
            filename = f"{etiqueta}_dibujado_{timestamp}.png"
            filepath = os.path.join(label_folder, filename)
            
            img.save(filepath)
            self.log(f" Dibujo guardado como: {filename}")
            self.limpiar_canvas()
            
        except Exception as e:
            self.log(f" Error guardando dibujo: {str(e)}")

    
    # --- MÉTODOS DE ENTRENAMIENTO ---
    def iniciar_entrenamiento(self):
        """Inicia el proceso de entrenamiento del modelo"""
        try:
            epochs = int(self.epochs_entry.get())
            self.btn_entrenar.config(state='disabled')
            
            self.log("🚀 Iniciando entrenamiento...")
            self.log(f"📊 Épocas: {epochs}")
            self.log("⏳ Esto puede tomar unos momentos...")
            
            # Ejecutar entrenamiento
            resultado = self.model_manager.entrenar_modelo(epochs=epochs)
            
            self.log(" ¡Entrenamiento completado!")
            self.log(f" Pérdida final: {resultado['final_loss']:.4f}")
            self.log(f" Precisión final: {resultado['final_accuracy']:.4f}")
            
            # Guardar modelo
            self.model_manager.guardar_modelo()
            
        except Exception as e:
            self.log(f" Error en entrenamiento: {str(e)}")
        finally:
            self.btn_entrenar.config(state='normal')
    
    def volver_principal(self):
        """Vuelve a la ventana principal"""
        self.root.destroy()
        self.ventana_principal.deiconify()
    
    def log(self, mensaje: str):
        """Agrega un mensaje al área de progreso"""
        self.progreso_text.insert(tk.END, f"{mensaje}\n")
        self.progreso_text.see(tk.END)
        self.root.update()