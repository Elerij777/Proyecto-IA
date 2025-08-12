import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
from imutils import paths
import os
from sklearn.svm import LinearSVC

# ============================
#   ENTRENAMIENTO DEL SVM
# ============================
dim = (50, 50)  # tamaño reducido para acelerar
interpolation = cv2.INTER_AREA

def cargarImagen(ruta):
    img = cv2.imread(ruta)
    img = cv2.resize(img, dim, interpolation=interpolation)
    return img

def extraerColores(img):
    (B, G, R) = cv2.split(img)
    # Convertir a HSV para mejor detección de colores, especialmente el rojo
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # El rojo está en los extremos del canal H, así que calculamos la media considerando ambos extremos
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    # También agregamos la media de los canales RGB para robustez
    b, g, r = cv2.split(img)
    return np.array([mean_h, mean_s, mean_v, np.mean(r), np.mean(g), np.mean(b)])

x, y = [], []
for ruta in paths.list_images("./entrenamiento"):  # dataset de entrenamiento
    img = cargarImagen(ruta)
    caracteristicas = extraerColores(img)
    etiqueta = ruta.split(os.path.sep)[-2]  # nombre carpeta = clase
    # Excluir la carpeta "captura" del entrenamiento
    if etiqueta.lower() != "captura":
        x.append(caracteristicas)
        y.append(etiqueta)

x = np.asarray(x)
model_color = LinearSVC(C=100.0, random_state=1, max_iter=1000)
model_color.fit(x, y)
print("[INFO] Modelo SVM (HSV+RGB) entrenado con", len(y), "ejemplos.")

# ============================
#   APLICACIÓN CON YOLO + SVM
# ============================
class ImageSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Objetos + Colores")
        self.root.geometry("750x600")
        self.root.configure(bg="#f0f0f0")

        ttk.Label(root, text="Detector de Objetos y Colores", font=("Helvetica", 18)).pack(pady=20)

        # Frame principal horizontal
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Frame de la imagen a la izquierda
        self.image_frame = tk.Frame(self.main_frame, bg="#ffffff", relief="groove", width=740, height=480)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack(expand=True)
        self.button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        button_style = {"width": 20}
        ttk.Button(self.button_frame, text="Seleccionar Imagen", command=self.seleccionar_imagen, **button_style).pack(pady=10, anchor="n")

        #Descarguen droidcam e instalenlo en la pc para usar la opción de DroidCam tiene que estar conectada
        ttk.Label(self.button_frame, text="Cámara:", background="#f0f0f0").pack(pady=(10,5), anchor="n")
        self.camera_var = tk.StringVar(value="Laptop")
        camera_frame = tk.Frame(self.button_frame, bg="#f0f0f0")
        camera_frame.pack(pady=5, anchor="n")
        ttk.Radiobutton(camera_frame, text="Laptop", variable=self.camera_var, value="Laptop").pack(anchor="w")
        ttk.Radiobutton(camera_frame, text="DroidCam", variable=self.camera_var, value="DroidCam").pack(anchor="w")
        
        ttk.Button(self.button_frame, text="Abrir Cámara", command=self.abrir_camara, **button_style).pack(pady=10, anchor="n")
        ttk.Button(self.button_frame, text="Detectar", command=self.detectar_objetos, state="normal", **button_style).pack(pady=5, anchor="n")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Helvetica", 12), padding=10)

        self.current_image_path = None
        self.model_yolo = YOLO("yolov8n.pt")  # Modelo YOLO

    def abrir_camara(self):
        camera_choice = self.camera_var.get()
        
        if camera_choice == "Laptop":
            camera_index = 0
            print("Usando cámara de la laptop (índice 0)")
        else:  # DroidCam
            camera_index = 1
            print("Usando DroidCam (índice 1)")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"No se pudo abrir la cámara {camera_choice} (índice {camera_index}).")
            return
            return
        cv2.namedWindow("Presiona ESPACIO para capturar, ESC para salir")
        frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Presiona ESPACIO para capturar, ESC para salir", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: 
                frame = None
                break
            elif key == 32:  
                break
        cap.release()
        cv2.destroyAllWindows()
        if frame is not None:
            import os, time
            save_dir = os.path.join("entrenamiento", "captura")
            os.makedirs(save_dir, exist_ok=True)
            import datetime
            filename = f"captura_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            temp_path = os.path.join(save_dir, filename)
            cv2.imwrite(temp_path, frame)
            time.sleep(0.1)
            img_bgr = cv2.imread(temp_path)
            if img_bgr is None:
                print("Error: la imagen capturada no se pudo leer correctamente.")
                return
            self.current_image_path = temp_path
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img.thumbnail((740, 480))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
        else:
            print("No se pudo capturar la imagen.")

    def seleccionar_imagen(self):
        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.current_image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((740, 480))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

    def detectar_objetos(self):
        if not self.current_image_path:
            return

        img_bgr = cv2.imread(self.current_image_path)
        #MJORA DE CONTRASTE PARA DETECCION DE COLOR
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=10)
        results = self.model_yolo(img_bgr)[0]

        found_object = False
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue  
            found_object = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_obj = self.model_yolo.names[int(box.cls[0])]
            pad = 5
            h, w = img_bgr.shape[:2]
            x1p = max(x1 - pad, 0)
            y1p = max(y1 - pad, 0)
            x2p = min(x2 + pad, w)
            y2p = min(y2 + pad, h)
            obj_crop = img_bgr[y1p:y2p, x1p:x2p]
            if obj_crop.size > 0:
                caracteristicas = extraerColores(cv2.resize(obj_crop, dim))
                color_pred = model_color.predict([caracteristicas])[0]
            else:
                color_pred = "Desconocido"

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{label_obj} - {color_pred} ({conf:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        if not found_object:
            #si no detecta YOLO el objeto pues busca el color dominante
            caracteristicas = extraerColores(cv2.resize(img_bgr, dim))
            color_pred = model_color.predict([caracteristicas])[0]
            cv2.putText(img_bgr, f"Color dominante: {color_pred}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 2)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((740, 480))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()
