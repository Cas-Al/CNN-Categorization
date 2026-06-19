
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from cnn_scratch import CNN, softmax

#  Configuration (must match train.py) 
MODEL_SAVE       = "prisma_cnn.pkl"
CLASS_MAPPING    = "class_mapping.pkl"
IMG_SIZE         = 32
FALLBACK_CLASSES = ["comedy", "fashion", "gaming", "music"]



def load_class_names():
    if os.path.exists(CLASS_MAPPING):
        with open(CLASS_MAPPING, "rb") as f:
            return pickle.load(f)
    print("[WARNING] class_mapping.pkl not found — using default order.")
    return FALLBACK_CLASSES


def preprocess_image(image_path):
    
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, np.newaxis, :, :]   # (1, 1, 32, 32)


def predict_image(image_path):
    #  1. Check files exist 
    if not os.path.exists(MODEL_SAVE):
        print(f"[ERROR] Model '{MODEL_SAVE}' not found. Run train.py first.")
        return
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    #  2. Load model 
    class_names = load_class_names()
    num_classes = len(class_names)

    model = CNN(num_classes=num_classes, input_channels=1, img_size=IMG_SIZE)
    model.load(MODEL_SAVE)

    #  3. Preprocess & predict 
    X = preprocess_image(image_path)
    logits = model.forward(X)
    probs  = softmax(logits)[0]   # (num_classes,)

    predicted_idx   = np.argmax(probs)
    predicted_class = class_names[predicted_idx]
    confidence      = probs[predicted_idx] * 100

    #  4. Print results 
    print("\n===== Prediction Result =====")
    print(f"Image      : {image_path}")
    print(f"Prediction : {predicted_class.upper()}")
    print(f"Confidence : {confidence:.2f}%")
    print("\nAll class probabilities:")
    for cat, prob in zip(class_names, probs):
        bar = "█" * int(prob * 40)
        print(f"  {cat:<15} {prob * 100:>6.2f}%  {bar}")

    #  5. Show image 
    img_display = mpimg.imread(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_display, cmap="gray" if img_display.ndim == 2 else None)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class}  ({confidence:.1f}%)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("prediction_output.png", dpi=150)
    plt.show()
    print("Output saved → prediction_output.png")


#  Entry point 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage  : python predict.py <image_path>")
        print("Example: python predict.py photo.jpg")
        sys.exit(1)
    predict_image(sys.argv[1])
