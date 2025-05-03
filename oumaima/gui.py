import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

# Load your trained model
model = YOLO('best.pt')

# Cancer descriptions
cancer_descriptions = {
    "Basal Cell Carcinoma": "Basal Cell Carcinoma is a common skin cancer that often appears as a slightly transparent bump on the skin. It's caused by prolonged sun exposure.",
    "Benign Keratosis": "Benign Keratosis is a non-cancerous skin growth often resembling warts or moles, commonly caused by aging or sun exposure.",
    "Melanoma": "Melanoma is a serious form of skin cancer that develops in melanocytes. It can spread quickly if not treated early.",
    "Nevus": "Nevus is a benign growth, commonly known as a mole, which is usually harmless but requires monitoring for changes."
}

def open_image():
    # Open file dialog to select an image
    filepath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if not filepath:
        return

    # Run inference
    results = model(filepath)

    # Display original image without prediction overlays
    image = Image.open(filepath).convert("RGB")
    image.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(image)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Extract prediction
    probs = results[0].probs
    top1 = probs.top1
    confidence = probs.top1conf.item()
    class_name = results[0].names[top1]
    true_class = os.path.basename(os.path.dirname(filepath))

    # Get cancer description
    description = cancer_descriptions.get(class_name, "No description available.")

    # Display result
    result_text = (
        f"True Class: {true_class}\n"
        f"Predicted Class: {class_name}\n"
        f"Confidence:      {confidence:.2%}\n\n"
        f"{description}"
    )
    update_results(result_text)

# Function to update results in the results box
def update_results(result_text):
    results_box_label.config(text=result_text)

# GUI setup
root = tk.Tk()
root.title("Skin Cancer Detection GUI")
root.geometry("900x700")
root.configure(bg="#f5f5f5")  # Soft background
root.resizable(False, False)
root.iconbitmap("icon.ico")  # Set the icon if available


# Add university logo
logo_image = Image.open("university_logo.png").resize((180, 80), Image.Resampling.LANCZOS)
logo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(root, image=logo, bg="#f5f5f5")
logo_label.pack(pady=10)

# Title Label
title_label = tk.Label(
    root, text="Skin Cancer Detection", 
    font=("Helvetica", 24, "bold"), fg="#1e1e2d", bg="#ffffff"
)
title_label.pack(pady=20)

# Upload Button
style = ttk.Style()
style.configure("Rounded.TButton", font=("Helvetica", 12), padding=10)
upload_button = ttk.Button(root, text="Upload Image", command=open_image, style="Rounded.TButton")
upload_button.pack(pady=10)

# Main Frame for Image and Results
main_frame = tk.Frame(root, bg="#ffffff", relief="flat", bd=2)
main_frame.pack(pady=20, padx=20, expand=True, fill=tk.BOTH)

# Left Frame for Image
left_frame = tk.Frame(main_frame, bg="#f9f9f9", bd=0, relief="ridge")
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

image_label = tk.Label(left_frame, bg="#e3f2fd", width=400, height=400)
image_label.pack()

# Right Frame for Results
right_frame = tk.Frame(main_frame, bg="#f9f9f9", bd=0, relief="ridge")
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill=tk.BOTH)

# Results Box
results_box_frame = tk.Frame(
    right_frame, bg="#ffffff", bd=2, relief="solid", padx=10, pady=10
)
results_box_frame.pack(padx=5, pady=5, expand=False, fill=tk.BOTH)

results_box_label = tk.Label(
    results_box_frame, text="Upload an image to see results", 
    font=("Helvetica", 14), bg="#ffffff", fg="#1e1e2d", 
    justify=tk.LEFT, wraplength=300
)
results_box_label.pack()

# Footer
footer_frame = tk.Frame(root, bg="#f5f5f5")
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
footer_label = tk.Label(
    footer_frame, text="Powered by YOLOv8 and Tkinter\nProject Founders: Khalil El Amraoui, John Doe, Jane Smith", 
    font=("Helvetica", 10), bg="#f5f5f5", fg="#616161"
)
footer_label.pack()

# Run the GUI
root.mainloop()
