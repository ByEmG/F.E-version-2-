import tkinter as tk
from sup import run_supervised
from unsupervised import run_unsupervised
from expression_matcher import compare_uploaded_with_webcam

# Initialize GUI window
root = tk.Tk()
root.title("Facial Emotion Recognition - Unified System")
root.geometry("480x450")


# Title
label = tk.Label(root, text="Gerard's Facial Emotion Recognition App ", font=("Helvetica", 18, "bold"), fg="#000080")
label.pack(pady=20)

# Supervised button
btn_supervised = tk.Button(
    root,
    text="Run Real-Time Supervised Emotion Recognition 📸 ",
    command=run_supervised,
    font=("Helvetica", 13),
    bg="Black",
    fg="Black",
    activebackground="#1C2541",
    activeforeground="Black",
    wraplength=350,
    width=38,
    height=3,
    bd=0
)
btn_supervised.pack(pady=10)

# Unsupervised button
btn_unsupervised = tk.Button(
    root,
    text="Run Unsupervised Emotion Clustering (K-Means)📊",
    command=run_unsupervised,
    font=("Helvetica", 13),
    bg="#000000",
    fg="Black",
    activebackground="#1C2541",
    activeforeground="white",
    wraplength=350,
    width=38,
    height=3,
    bd=0
)
btn_unsupervised.pack(pady=10)

# Match face with uploaded image
btn_match = tk.Button(
    root,
    text="Match Your Expression with an Uploaded Image📷",
    command=compare_uploaded_with_webcam,
    font=("Helvetica", 13),
    bg="#000000",
    fg="Black",
    activebackground="#1C2541",
    activeforeground="white",
    wraplength=350,
    width=38,
    height=3,
    bd=0
)
btn_match.pack(pady=10)

# Exit button
btn_exit = tk.Button(
    root,
    text="Exit❌",
    command=root.quit,
    font=("Helvetica", 13),
    bg="#000000",
    fg="Black",
    width=20,
    height=2,
    bd=0
)
btn_exit.pack(pady=30)

root.mainloop()