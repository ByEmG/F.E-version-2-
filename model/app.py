# main_app.py
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from sup import run_supervised
from unsupervised import run_unsupervised
from expression_matcher import compare_uploaded_with_webcam

def launch_supervised():
    messagebox.showinfo("Launching", "Starting Supervised Emotion Recognition...")
    run_supervised()

def launch_unsupervised():
    messagebox.showinfo("Launching", "Starting Unsupervised Emotion Clustering...")
    run_unsupervised()

def upload_and_compare():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if not path:
        return
    compare_uploaded_with_webcam(path)
# GUI Setup
root = tk.Tk()
root.title("Facial Emotion Recognition")
root.geometry("400x300")

tk.Label(root, text="Welcome to Gerard's Face Emotion recognition to start Choose a mode", font=("Helvetica", 16), wraplength=300).pack(pady=20)
tk.Button(root, text="Supervised (CNN)", width=30, command=launch_supervised).pack(pady=10)
tk.Button(root, text="Unsupervised (K-Means)", width=30, command=launch_unsupervised).pack(pady=10)
btn4 = tk.Button(root, text="Match Your Expression with an Image 🏞️", font=("Helvetica", 14), bg="#9C27B0", fg="Black", width=35, command=compare_uploaded_with_webcam)
btn4.pack(pady=10)
root.mainloop()
