
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

from cnn_scratch import CNN, cross_entropy_loss, softmax

#  Configuration 
DATASET_DIR  = "dataset"
IMG_SIZE     = 32              # resize all images to 32×32
CATEGORIES   = ["comedy", "fashion", "gaming", "music"]   
EPOCHS       = 25
BATCH_SIZE   = 16
LEARNING_RATE = 0.01
LR_DECAY     = 0.95            # multiply LR by this every epoch
MODEL_SAVE   = "prisma_cnn.pkl"

# STEP 1 — Image Loading

def load_image(path):
    
    img = Image.open(path).convert("L")               # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0     # normalize
    return arr[np.newaxis, :, :]                       # (1, H, W)


def load_dataset(split="train"):
    X_list, y_list = [], []
    split_dir = os.path.join(DATASET_DIR, split)

    # Sort categories so label indices are consistent
    found_cats = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    print(f"[{split}] Found categories: {found_cats}")

    for label_idx, cat in enumerate(found_cats):
        cat_dir = os.path.join(split_dir, cat)
        files = [
            f for f in os.listdir(cat_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"  {cat}: {len(files)} images (label {label_idx})")

        for fname in files:
            try:
                img = load_image(os.path.join(cat_dir, fname))
                X_list.append(img)
                y_list.append(label_idx)
            except Exception as e:
                print(f"    [SKIP] {fname}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, found_cats



# STEP 2 — Accuracy Computation (for evaluation)

def compute_accuracy(model, X, y, batch_size=32):
    
    correct = 0
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        preds, _ = model.predict(X_batch)
        correct += (preds == y_batch).sum()
    return correct / len(y)


# STEP 3 — Training Loop

def train():
    # Load data
    print("Loading training data...")
    X_train, y_train, class_names = load_dataset("train")
    print("\nLoading test data...")
    X_test, y_test, _ = load_dataset("test")

    print(f"\nTraining set : {X_train.shape[0]} images")
    print(f"Test set     : {X_test.shape[0]} images")
    print(f"Image shape  : {X_train.shape[1:]}")
    print(f"Classes      : {class_names}\n")

    # Save class mapping so predict.py knows the order
    with open("class_mapping.pkl", "wb") as f:
        pickle.dump(class_names, f)

    # Build model
    model = CNN(num_classes=len(class_names), input_channels=1, img_size=IMG_SIZE)

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [],  "test_acc": []
    }

    lr = LEARNING_RATE
    N = len(X_train)
    indices = np.arange(N)

    print("===== Training Started =====")
    print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {lr}\n")

    for epoch in range(1, EPOCHS + 1):
        # Shuffle training data each epoch
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch gradient descent
        for i in range(0, N, BATCH_SIZE):
            X_batch = X_train[i:i + BATCH_SIZE]
            y_batch = y_train[i:i + BATCH_SIZE]

            #  Forward pass 
            logits = model.forward(X_batch)

            #  Compute loss 
            loss, dlogits = cross_entropy_loss(logits, y_batch)
            epoch_loss += loss

            #  Backward pass 
            model.backward(dlogits)

            #   Update weights 
            model.update_weights(lr)

            num_batches += 1

        # Decay learning rate
        lr *= LR_DECAY

        # Evaluate
        avg_loss  = epoch_loss / num_batches
        train_acc = compute_accuracy(model, X_train, y_train)
        test_acc  = compute_accuracy(model, X_test, y_test)

        # Test loss
        test_logits = model.forward(X_test)
        test_loss, _ = cross_entropy_loss(test_logits, y_test)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch:>3}/{EPOCHS}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Train Acc: {train_acc*100:.1f}%  |  "
              f"Test Acc: {test_acc*100:.1f}%  |  "
              f"LR: {lr:.6f}")

    # Save model
    model.save(MODEL_SAVE)

    # Final evaluation
    print("\n===== Final Results =====")
    final_acc = compute_accuracy(model, X_test, y_test)
    print(f"Test Accuracy : {final_acc * 100:.2f}%")

    # Plot curves
    plot_history(history)
    return model, history


# STEP 4 — Plot Training Curves
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_acc"]) + 1)

    ax1.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy")
    ax1.plot(epochs, [a * 100 for a in history["test_acc"]],  label="Test Accuracy")
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_loss"], label="Train Loss")
    ax2.plot(epochs, history["test_loss"],  label="Test Loss")
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved → training_curves.png")


#  Entry point 
if __name__ == "__main__":
    train()
