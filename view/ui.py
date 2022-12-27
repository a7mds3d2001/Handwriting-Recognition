from keras.models import load_model
from PIL import ImageGrab
from tkinter import *
import numpy as np
import win32gui

# Import Model to Use In File
model = load_model('hand_writing_recognition.h5')

# Predict Digit
def predict_digit(img):
    # Resize Image To 28*28 Pixels
    img = img.resize((28, 28))

    # Convert RGB To GrayScale
    img = img.convert('L')
    img = np.array(img)

    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img/255.0

    # predicting the class
    res = model.predict([img])[0]

    return np.argmax(res), max(res)

class Root:
    
    def begin(self):
        self.splash_root = Tk()
        # Window Settings
        self.splash_root.title('Splash Screen')
        self.splash_root.config(background='#1d3557')
        self.splash_root.geometry("500x250+500+250")
        self.splash_root.overrideredirect(True)

        # Splash Title 
        self.splashLabel = Label(
            self.splash_root,
            text = "Hand Writing Recognition",
            foreground = '#f1faee',
            background = '#1d3557',
            font = ("Helvetica", 30)
        )
        self.splashLabel.pack(pady = 100)

        # After 3s Navigate to MainWindow
        self.splash_root.after(500, self.main_window)
        self.splash_root.mainloop()
    
    def main_window(self):
        # Delete Window SplashRoot
        self.splash_root.destroy()

        # Window Settings
        self.root = Tk()
        self.root.title('Hand Writing Recognition')
        self.root.config(background='#1d3557')
        # self.root.geometry("1024x800+320+0")
        self.root.state('zoomed')

        # getting screen's height in pixels
        height = self.root.winfo_screenheight()
        
        # getting screen's width in pixels
        width = self.root.winfo_screenwidth()

        # Drawing Space For Digit
        self.canvas = Canvas(
            self.root,
            width = width * 0.54,
            height = height * 0.7,
            bg = "#1d3557",
            bd=5,
            cursor = "dot"
        )

        # Status of Line
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        # Canvas Grid -> 0, 0
        self.canvas.grid(row = 0, column = 0, pady = height * 0.05, padx = height * 0.05)
        
        # Label Before PredictDigit 
        self.label = Label(
            self.root,
            text = "Draw Digit Now..",
            background='#1d3557',
            foreground='White',
            font = ("Helvetica", 30, 'bold')
        )

        # Label Grid -> 0, 1
        self.label.grid(row = 0, column = 1, pady = 50, padx = 50)

        # Clear Drawing
        self.btn_clear = Button(
            self.root,
            text = "Clear",
            fg = '#f1faee',
            activeforeground = '#f1faee',
            bg = 'red',
            activebackground = 'red',
            width = 20,
            height = 2,
            bd = 0,
            font = ("Helvetica", 14, 'bold'),        
            command = self.clear_all
        )

        # Clear Drawing Grid -> 1, 0
        self.btn_clear.grid(row = 1, column = 0, pady = 2)
        
        # Classify Driwing 
        self.btn_classify = Button(
            self.root,
            text = "Recognise",
            fg = '#f1faee',
            activeforeground = '#f1faee',
            bg = 'green',
            activebackground = 'green',
            width = 20,
            height = 2,
            bd = 0,
            font = ("Helvetica", 14, 'bold'),    
            command = self.classify_handwriting
        )

        # Classify Drawing Grid -> 1, 1
        self.btn_classify.grid(row = 1, column = 1, pady = 2, padx = 2)

        self.root.mainloop()

    # CLEAR DRAWING SPACE
    def clear_all(self):
        self.label.configure(text='Draw Digit Now..')
        self.x = 0
        self.y = 0
        self.canvas.delete("all")

    # Drawing Digit
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y

        # Size Of Point
        r = 15

        # Point
        self.canvas.create_oval(
            self.x - r,
            self.y - r,
            self.x + r,
            self.y + r,
            fill='white'
        )

    def classify_handwriting(self):
        if self.x != 0 or self.y ==0 :
            # Get The Handle Of The Canvas
            HWND = self.canvas.winfo_id()

            # Get The Coordinate Of The Canvas
            rect = win32gui.GetWindowRect(HWND)
            im = ImageGrab.grab(rect)

            # Result Of Processing Digit
            digit, acc = predict_digit(im)

            # Display Result
            self.label.configure(text=str(digit)+', ' + str(int(acc*100))+'%',)