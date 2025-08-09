import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image






class ImageSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Objetos con YOLOv8")
        self.root.geometry("700x550")
        self.root.configure(bg="#f0f0f0")

        self.label_title = ttk.Label(root, text="Selecciona una Imagen", font=("Helvetica", 18))
        self.label_title.pack(pady=20)

        self.image_frame = tk.Frame(root, bg="#ffffff", relief="groove", width=500, height=350)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack(expand=True)

        self.btn_select = ttk.Button(root, text="Seleccionar Imagen", command=self.seleccionar_imagen)
        self.btn_select.pack(pady=10)

        self.btn_detect = ttk.Button(root, text="Detectar Objetos", command=self.detectar_objetos, state="disabled")
        self.btn_detect.pack(pady=5)

        self.current_image_path = None

        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Helvetica", 12), padding=10)

        # Cargar modelo YOLOv8
        self.model = YOLO("yolov8n.pt")  # usa la versión más ligera para velocidad

    def seleccionar_imagen(self):
        file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
        if file_path:
            self.current_image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((500, 350))
            img_tk = ImageTk.PhotoImage(img)

            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
            self.btn_detect.config(state="normal")

            plot_colors(colors)


    def detectar_objetos(self):
        if not self.current_image_path:
            return

        img_bgr = cv2.imread(self.current_image_path)
        results = self.model(img_bgr)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((500, 350))
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()  