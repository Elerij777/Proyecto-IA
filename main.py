import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
from imutils import paths
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# ============================
#   ENTRENAMIENTO DEL SVM
# ============================
dim = (50, 50)
interpolation = cv2.INTER_AREA

def cargarImagen(ruta):
    img = cv2.imread(ruta)
    img = cv2.resize(img, dim, interpolation=interpolation)
    return img

def extraerColores(img):
    (B, G, R) = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    b, g, r = cv2.split(img)
    return np.array([mean_h, mean_s, mean_v, np.mean(r), np.mean(g), np.mean(b)])

x, y = [], []
for ruta in paths.list_images("./entrenamiento"):
    img = cargarImagen(ruta)
    caracteristicas = extraerColores(img)
    etiqueta = ruta.split(os.path.sep)[-2]
    if etiqueta.lower() != "captura":
        x.append(caracteristicas)
        y.append(etiqueta)

x = np.asarray(x)
model_color = LinearSVC(C=100.0, random_state=1, max_iter=1000)
model_color.fit(x, y)
print("[INFO] Modelo SVM (HSV+RGB) entrenado con", len(y), "ejemplos.")

# ============================
#   RESULTADOS DEL ENTRENAMIENTO
# ============================
tam_dataset = len(y)
clases_colores = sorted(set(y))

# Calcular precisión estimada con división entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model_temp = LinearSVC(C=100.0, random_state=1, max_iter=1000)
model_temp.fit(X_train, y_train)
precision_test = model_temp.score(X_test, y_test) * 100

print(f"Tamaño del dataset: {tam_dataset} imágenes.")
print(f"Clases de colores: {clases_colores}")
print(f"Precisión estimada: {precision_test:.2f}% en pruebas internas.")
print("==============================================\n")

# ============================
#   APLICACIÓN CON YOLO + SVM
# ============================
class ImageSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Objetos + Colores")
        self.root.geometry("800x650")
        self.root.configure(bg="#e9ecef")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton",
                        font=("Segoe UI", 12, "bold"),
                        padding=10,
                        background="#4f8cff",
                        foreground="#fff",
                        borderwidth=0,
                        focusthickness=3,
                        focuscolor="#0056b3")
        style.map("TButton",
                  background=[('active', '#0056b3')],
                  foreground=[('active', '#fff')])
        style.configure("TLabel",
                        font=("Segoe UI", 13),
                        background="#e9ecef",
                        foreground="#222")
        title_frame = tk.Frame(self.root, bg="#e9ecef")
        title_frame.pack(pady=(18, 5))
        ttk.Label(title_frame, text="Detector de Objetos y Colores", font=("Segoe UI", 22, "bold"),
                  background="#e9ecef", foreground="#4f8cff").pack()
        ttk.Label(title_frame, text="Proyecto IA", font=("Segoe UI", 12),
                  background="#e9ecef", foreground="#888").pack()

        # Frame principal
        self.main_frame = tk.Frame(self.root, bg="#e9ecef")
        self.main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.image_frame = tk.Frame(self.main_frame, bg="#fff", relief="groove", bd=2,
                                    width=740, height=480, highlightbackground="#4f8cff", highlightthickness=2)
        self.image_frame.pack(side=tk.LEFT, padx=(30, 10), pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#fff")
        self.image_label.pack(expand=True)

        self.button_frame = tk.Frame(self.main_frame, bg="#e9ecef")
        self.button_frame.pack(side=tk.LEFT, padx=(10, 30), pady=10, fill=tk.Y)

        button_style = {"width": 20}
        ttk.Button(self.button_frame, text="Seleccionar Imagen", command=self.seleccionar_imagen,
                   **button_style).pack(pady=(10, 18), anchor="n")

        cam_label = ttk.Label(self.button_frame, text="Cámara:", background="#e9ecef",
                              font=("Segoe UI", 12, "bold"), foreground="#4f8cff")
        cam_label.pack(pady=(10, 5), anchor="n")

        self.camera_var = tk.StringVar(value="Laptop")
        camera_frame = tk.Frame(self.button_frame, bg="#e9ecef")
        camera_frame.pack(pady=5, anchor="n")
        ttk.Radiobutton(camera_frame, text="Laptop", variable=self.camera_var, value="Laptop").pack(anchor="w")
        ttk.Radiobutton(camera_frame, text="DroidCam", variable=self.camera_var, value="DroidCam").pack(anchor="w")

        ttk.Button(self.button_frame, text="Abrir Cámara", command=self.abrir_camara,
                   **button_style).pack(pady=(18, 10), anchor="n")
        ttk.Button(self.button_frame, text="Detectar", command=self.detectar_objetos,
                   state="normal", **button_style).pack(pady=5, anchor="n")

        footer = ttk.Label(self.root, text="Desarrollado por Proyecto IA", font=("Segoe UI", 10),
                           background="#e9ecef", foreground="#aaa")
        footer.pack(side=tk.BOTTOM, pady=8)

        self.current_image_path = None
        self.model_yolo = YOLO("yolov8n.pt")

    def abrir_camara(self):
        camera_choice = self.camera_var.get()
        camera_index = 0 if camera_choice == "Laptop" else 1
        print(f"Usando {'cámara de la laptop' if camera_index == 0 else 'DroidCam'} (índice {camera_index})")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"No se pudo abrir la cámara {camera_choice} (índice {camera_index}).")
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
            import time, datetime
            save_dir = os.path.join("entrenamiento", "captura")
            os.makedirs(save_dir, exist_ok=True)
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
            x1p, y1p = max(x1 - pad, 0), max(y1 - pad, 0)
            x2p, y2p = min(x2 + pad, w), min(y2 + pad, h)
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
