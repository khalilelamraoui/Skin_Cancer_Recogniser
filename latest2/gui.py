import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load your trained model
#model = YOLO('./runs/detect/train7/weights/best.pt')
model = YOLO('model.pt')

def open_image():
    # Open file dialog to select an image
    filepath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if not filepath:
        return
    
    # Run inference
    results = model(filepath)
    results_image = results[0].plot()  # Annotated image
    
    # Display the image in GUI
    image = Image.fromarray(results_image)
    image.thumbnail((400, 400))  # Resize for display
    img_tk = ImageTk.PhotoImage(image)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    
    # Show prediction results
    result_text = ""
    for detection in results[0].boxes.data:
        class_id = int(detection[5])  # Class ID
        confidence = float(detection[4])  # Confidence score
        class_name = results[0].names[class_id]  # Class name
        result_text += f"{class_name}: {confidence:.2%}\n"
    
    if not result_text:
        result_text = "No detections found."
    
    result_label.config(text=result_text)

# GUI setup
root = tk.Tk()
root.title("Cancer Detection GUI")
root.geometry("800x600")
root.config(bg="#F57C00")  # Orange background

# Title Label
title_label = tk.Label(root, text="Skin Cancer Detection", font=("Helvetica", 24, "bold"), fg="white", bg="#0277BD")  # Blue color
title_label.pack(pady=10, fill=tk.X)

# Upload Button
upload_button = ttk.Button(root, text="Upload Image", command=open_image)
upload_button.pack(pady=10)

# Main Frame for Image and Results
main_frame = tk.Frame(root, bg="#F57C00")
main_frame.pack(pady=10, expand=True, fill=tk.BOTH)

# Left Frame for Image
left_frame = tk.Frame(main_frame, bg="#F57C00")
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

image_label = tk.Label(left_frame, bg="#E3F2FD", width=400, height=400)  # Light blue background
image_label.pack()

# Right Frame for Results
right_frame = tk.Frame(main_frame, bg="#F57C00")
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

result_label = tk.Label(right_frame, text="Upload an image to see results", font=("Helvetica", 14), bg="#F57C00", fg="white", justify=tk.LEFT, wraplength=300)
result_label.pack()

# Footer
footer_label = tk.Label(root, text="Powered by YOLOv8 and Tkinter", font=("Helvetica", 10), bg="#F57C00", fg="white")
footer_label.pack(side=tk.BOTTOM, pady=10)

# Run the GUI
root.mainloop()