import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

# Load your trained model
model = YOLO('best.pt')
# model = YOLO('best.pt')

# def open_image():
#     # Open file dialog to select an image
#     filepath = filedialog.askopenfilename(
#         filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
#     )
#     if not filepath:
#         return
    
#     # Run inference
#     results = model(filepath)
#     results_image = results[0].plot()  # Annotated image
    
#     # Display the image in GUI
#     image = Image.fromarray(results_image)
#     image.thumbnail((400, 400))  # Resize for display
#     img_tk = ImageTk.PhotoImage(image)
#     image_label.config(image=img_tk)
#     image_label.image = img_tk
    
#     # Show prediction results
#     result_text = ""
#     for detection in results[0].boxes.data:
#         class_id = int(detection[5])  # Class ID
#         confidence = float(detection[4])  # Confidence score
#         class_name = results[0].names[class_id]  # Class name
#         result_text += f"{class_name}: {confidence:.2%}\n"
    
#     if not result_text:
#         result_text = "No detections found."
    
#     result_label.config(text=result_text)
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
    image.thumbnail((400, 400))  # Maintain original size
    img_tk = ImageTk.PhotoImage(image)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Extract prediction
    probs = results[0].probs
    top1 = probs.top1
    confidence = probs.top1conf.item()
    class_name = results[0].names[top1]

    # Get true class from folder name (e.g., path/to/test/ClassX/image.jpg → ClassX)
    true_class = os.path.basename(os.path.dirname(filepath))

    # Update table with results
    for child in result_table.get_children():
        result_table.delete(child)

    result_table.insert("", "end", values=(true_class, class_name, f"{confidence:.2%}"))

    # Check if prediction is correct
    if true_class == class_name:
        result_message.config(text="Success: Prediction matches the true class!", fg="green")
    else:
        result_message.config(text="Error: Prediction does not match the true class.", fg="red")

# GUI setup
root = tk.Tk()
root.title("Cancer Detection GUI")
root.geometry("900x750")
root.config(bg="#FFF7E6")  # Creamy background

# Title Label
title_label = tk.Label(root, text="Skin Cancer Detection", font=("Helvetica", 28, "bold"), fg="white", bg="#0277BD")  # Blue color
title_label.pack(pady=10, fill=tk.X)

# Upload Button
style = ttk.Style()
style.configure("Custom.TButton", 
                 font=("Helvetica", 14), 
                 padding=10, 
                 foreground="#F57C00", 
                 borderwidth=2, 
                 relief="raised")
style.map("Custom.TButton",
          bordercolor=[("!disabled", "#F57C00")])
upload_button = ttk.Button(root, text="Upload Image", command=open_image, style="Custom.TButton")
upload_button.pack(pady=20)

# Main Frame for Image and Results
main_frame = tk.Frame(root, bg="#FFF7E6")
main_frame.pack(pady=10, expand=True, fill=tk.BOTH)

# Left Frame for Image
left_frame = tk.Frame(main_frame, bg="#FFF7E6")
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

image_label = tk.Label(left_frame, bg="#E3F2FD", width=400, height=400)  # Light blue background
image_label.pack()

# Right Frame for Results
right_frame = tk.Frame(main_frame, bg="#FFF7E6")
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create a style for the Treeview
style = ttk.Style()
style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))  # Font for headings
style.configure("Treeview", font=("Helvetica", 10, "bold"))  # Font for rows

# Result Table
columns = ("True Class", "Predicted Class", "Confidence")
result_table = ttk.Treeview(right_frame, columns=columns, show="headings", height=1)
result_table.heading("True Class", text="True Class")
result_table.heading("Predicted Class", text="Predicted Class")
result_table.heading("Confidence", text="Confidence")
result_table.column("True Class", width=150, anchor="center")
result_table.column("Predicted Class", width=150, anchor="center")
result_table.column("Confidence", width=150, anchor="center")
result_table.pack(pady=20)

# Add Message Label
result_message = tk.Label(right_frame, text="", font=("Helvetica", 16), bg="#FFF7E6")
result_message.pack(pady=10)

# Footer
footer_label = tk.Label(root, text="Developed by: El Amraoui Khalil & Tariq Mohammed", font=("Helvetica", 12, 'bold', 'italic'), bg="#FFF7E6", fg="#F57C00")
footer_label.pack(side=tk.BOTTOM, pady=10)

# Run the GUI
root.mainloop()