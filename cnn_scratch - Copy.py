import numpy as np
import pickle

# UTILITY: im2col and col2im
# Instead of slow nested loops for convolution, we use im2col:
# → "unroll" every filter-sized patch of the image into a column
# → then one big matrix multiplication replaces all the sliding window loops

def im2col(X, fh, fw, stride, pad):
    
    N, C, H, W = X.shape
    out_h = (H + 2 * pad - fh) // stride + 1
    out_w = (W + 2 * pad - fw) // stride + 1

    X_pad = np.pad(X, [(0,0), (0,0), (pad,pad), (pad,pad)], mode='constant')
    col = np.zeros((N, C, fh, fw, out_h, out_w), dtype=X.dtype)

    for y in range(fh):
        y_end = y + stride * out_h
        for x in range(fw):
            x_end = x + stride * out_w
            col[:, :, y, x, :, :] = X_pad[:, :, y:y_end:stride, x:x_end:stride]

    # Rearrange into (N*out_h*out_w, C*fh*fw)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, X_shape, fh, fw, stride, pad):
    
    N, C, H, W = X_shape
    out_h = (H + 2 * pad - fh) // stride + 1
    out_w = (W + 2 * pad - fw) // stride + 1

    col = col.reshape(N, out_h, out_w, C, fh, fw).transpose(0, 3, 4, 5, 1, 2)
    X_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)

    for y in range(fh):
        y_end = y + stride * out_h
        for x in range(fw):
            x_end = x + stride * out_w
            X_pad[:, :, y:y_end:stride, x:x_end:stride] += col[:, :, y, x, :, :]

    if pad == 0:
        return X_pad
    return X_pad[:, :, pad:-pad, pad:-pad]



# LAYER 1: Convolutional Layer
class ConvLayer:

    def __init__(self, num_filters, filter_size, input_channels, stride=1, padding=1):
        self.F  = num_filters
        self.fh = filter_size   
        self.fw = filter_size
        self.C  = input_channels
        self.stride  = stride
        self.padding = padding

        # He initialization: prevents vanishing/exploding gradients
        scale = np.sqrt(2.0 / (input_channels * filter_size * filter_size))
        self.W = np.random.randn(num_filters, input_channels, filter_size, filter_size).astype(np.float32) * scale
        self.b = np.zeros((num_filters, 1), dtype=np.float32)

        # Gradients (computed during backward pass)
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out_h = (H + 2 * self.padding - self.fh) // self.stride + 1
        out_w = (W + 2 * self.padding - self.fw) // self.stride + 1

        # im2col: unroll image patches into rows
        self.col = im2col(X, self.fh, self.fw, self.stride, self.padding)
        # W reshaped to (F, C*fh*fw) for matrix multiply
        self.W_col = self.W.reshape(self.F, -1)

        # Core computation: (N*out_h*out_w, C*fh*fw) × (C*fh*fw, F) = (N*out_h*out_w, F)
        out = self.col @ self.W_col.T + self.b.T

        # Reshape to (N, F, out_h, out_w)
        out = out.reshape(N, out_h, out_w, self.F).transpose(0, 3, 1, 2)
        return out.astype(np.float32)

    def backward(self, dout):
      
        N, F, out_h, out_w = dout.shape

        # Reshape dout to (N*out_h*out_w, F)
        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, self.F)

        # Gradient w.r.t. weights: how much each filter weight contributed to error
        self.dW = (dout_col.T @ self.col).reshape(self.W.shape)
        # Gradient w.r.t. biases: sum over all positions
        self.db = dout_col.sum(axis=0).reshape(self.b.shape)

        # Gradient w.r.t. input: needed to continue backprop to earlier layers
        dcol = dout_col @ self.W_col
        dX = col2im(dcol, self.X.shape, self.fh, self.fw, self.stride, self.padding)
        return dX.astype(np.float32)



# LAYER 2: ReLU Activation
class ReLULayer:
   

    def forward(self, X):
        self.mask = (X > 0)          # remember which values were positive
        return (X * self.mask).astype(np.float32)

    def backward(self, dout):
        # Gradient only flows through positions where input was > 0
        return (dout * self.mask).astype(np.float32)



# LAYER 3: Max Pooling
class MaxPoolLayer:
    

    def __init__(self, pool_h=2, pool_w=2, stride=2):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride

    def forward(self, X):
        
        self.X = X
        N, C, H, W = X.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1

        # Use im2col trick for pooling too
        col = im2col(X, self.pool_h, self.pool_w, self.stride, pad=0)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # Keep the index of the max value in each window (needed for backprop)
        self.arg_max = np.argmax(col, axis=1)
        out = col[np.arange(col.shape[0]), self.arg_max]
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out.astype(np.float32)

    def backward(self, dout):
        
        N, C, out_h, out_w = dout.shape
        pool_size = self.pool_h * self.pool_w

        dmax = np.zeros((dout.size, pool_size), dtype=np.float32)
        dout_flat = dout.transpose(0, 2, 3, 1).ravel()
        dmax[np.arange(self.arg_max.size), self.arg_max] = dout_flat

        dmax = dmax.reshape(N, out_h, out_w, C, pool_size)
        dcol = dmax.reshape(N * out_h * out_w * C, pool_size)
        dX = col2im(dcol, self.X.shape, self.pool_h, self.pool_w, self.stride, pad=0)
        return dX.astype(np.float32)



# LAYER 4: Flatten
class FlattenLayer:
    
    def forward(self, X):
        self.original_shape = X.shape
        return X.reshape(X.shape[0], -1).astype(np.float32)

    def backward(self, dout):
        return dout.reshape(self.original_shape).astype(np.float32)



# LAYER 5: Dense (Fully Connected) Layer

class DenseLayer:

    def __init__(self, input_size, output_size):
        # He initialization
        scale = np.sqrt(2.0 / input_size)
        self.W = np.random.randn(input_size, output_size).astype(np.float32) * scale
        self.b = np.zeros((1, output_size), dtype=np.float32)

        self.dW = None
        self.db = None

    def forward(self, X):
        
        self.X = X
        return (X @ self.W + self.b).astype(np.float32)

    def backward(self, dout):
        
        # Gradient w.r.t. weights and biases
        self.dW = (self.X.T @ dout).astype(np.float32)
        self.db = dout.sum(axis=0, keepdims=True).astype(np.float32)
        # Gradient w.r.t. input (passed to previous layer)
        return (dout @ self.W.T).astype(np.float32)



# LOSS: Softmax + Cross-Entropy (combined)

def softmax(x):
    
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits, labels):
    
    N = len(labels)
    probs = softmax(logits)

    # Loss: -log(probability of correct class)
    correct_probs = probs[np.arange(N), labels]
    loss = -np.log(correct_probs + 1e-8).mean()

    # Gradient of Softmax + CrossEntropy combined (elegant closed form)
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1   # subtract 1 from correct class
    dlogits /= N
    return loss, dlogits.astype(np.float32)



# THE CNN MODEL: Assembles all layers

class CNN:

    def __init__(self, num_classes=4, input_channels=1, img_size=32):
        # After 2 MaxPool layers (each halves size): img_size / 4
        pool_out = img_size // 4   # = 8 for 32×32 input
        flat_size = 16 * pool_out * pool_out  # 16 filters × 8 × 8 = 1024

        self.layers = [
            # ── Conv Block 1 ──────────────────────────────────────────────────
            ConvLayer(num_filters=8, filter_size=3, input_channels=input_channels, padding=1),
            ReLULayer(),
            MaxPoolLayer(2, 2, stride=2),   # 32×32 → 16×16

            # ── Conv Block 2 ──────────────────────────────────────────────────
            ConvLayer(num_filters=16, filter_size=3, input_channels=8, padding=1),
            ReLULayer(),
            MaxPoolLayer(2, 2, stride=2),   # 16×16 → 8×8

            # ── Classifier Head ───────────────────────────────────────────────
            FlattenLayer(),                  # 16 × 8 × 8 = 1024

            DenseLayer(flat_size, 128),
            ReLULayer(),

            DenseLayer(128, num_classes),    # final scores (logits)
        ]

    def forward(self, X):
        
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out   # logits (N, num_classes)

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_weights(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.dW is not None:
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

    def predict(self, X):
        logits = self.forward(X)
        probs  = softmax(logits)
        return np.argmax(probs, axis=1), probs

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                params[f'{i}_W'] = layer.W
                params[f'{i}_b'] = layer.b
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model saved → {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                layer.W = params[f'{i}_W']
                layer.b = params[f'{i}_b']
        print(f"Model loaded ← {path}")
