import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
import numpy as np
from tensorflow.keras.models import load_model

# Memuat model yang telah dilatih
model = load_model('mnist_trained_model.h5')

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack()

        # Mengubah ketebalan kuas
        self.brush_size = 20

        self.canvas.bind("<B1-Motion>", self.paint)

        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.button_recognize = tk.Button(root, text="Recognize", command=self.recognize)
        self.button_recognize.pack()

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=self.brush_size)

    def clear(self):
        self.canvas.delete("all")

    def recognize(self):
        # Menangkap gambar dari canvas
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
        
        # Menginversi warna gambar dan meresize
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image = np.array(image)
        
        # Normalisasi gambar
        image = image / 255.0
        image = image.reshape(1, 28, 28, 1)
        
        # Mengenali gambar menggunakan model yang telah dilatih
        prediction = model.predict(image)
        digit = np.argmax(prediction)
        
        # Menampilkan digit yang dikenali
        self.canvas.create_text(200, 200, text=str(digit), font=("Purisa", 48), fill='red')

# Menginisialisasi aplikasi Tkinter
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()