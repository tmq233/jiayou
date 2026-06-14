"""
Pure NumPy neural network building blocks for educational purposes.

Covers: forward/backward propagation, weight initialization strategies,
        LayerNorm vs BatchNorm, residual connections, Dropout.

All operations are vectorized. Dimension annotations follow every reshape/permute.
"""

import numpy as np


# ===================================================================
# Weight Initialization
# ===================================================================

def xavier_init(fan_in, fan_out, gain=1.0):
    """Xavier (Glorot) uniform initialization.

    Var(W) = 2 / (fan_in + fan_out)  —  preserves variance for Tanh/Sigmoid.
    """
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, (fan_in, fan_out))


def he_init(fan_in, fan_out, gain=1.0):
    """He (Kaiming) normal initialization.

    Var(W) = 2 / fan_in  —  compensates for ReLU's zero-half.
    """
    std = gain * np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


def orthogonal_init(shape, gain=1.0):
    """Orthogonal initialization via QR decomposition.

    Produces an (approximately) orthogonal matrix, preserving gradient
    norm during backprop — useful for PPO policy networks.
    """
    a = np.random.randn(*shape)
    if shape[0] > shape[1]:
        q, _ = np.linalg.qr(a)
    else:
        q, _ = np.linalg.qr(a.T)
        q = q.T
    return gain * q[:shape[0], :shape[1]]


# ===================================================================
# Layers
# ===================================================================

class Linear:
    """Fully-connected layer: y = x @ W + b."""

    def __init__(self, in_features, out_features, init='he'):
        self.in_features = in_features
        self.out_features = out_features

        if init == 'xavier':
            self.W = xavier_init(in_features, out_features)
        elif init == 'he':
            self.W = he_init(in_features, out_features)
        elif init == 'orthogonal':
            self.W = orthogonal_init((in_features, out_features))
        else:
            raise ValueError(f"Unknown init: {init}")

        self.b = np.zeros(out_features)
        self._cache = {}

    def forward(self, x):
        """x: [N, in_features] → y: [N, out_features]"""
        self._cache['x'] = x
        return x @ self.W + self.b  # [N, out_features]

    def backward(self, dout):
        """dout: [N, out_features] → dx: [N, in_features]"""
        x = self._cache['x']                    # [N, in_features]
        self._cache['dW'] = x.T @ dout          # [in_features, out_features]
        self._cache['db'] = dout.sum(axis=0)    # [out_features]
        return dout @ self.W.T                  # [N, in_features]

    def parameters(self):
        return {'W': self.W, 'b': self.b}

    def gradients(self):
        return {'W': self._cache['dW'], 'b': self._cache['db']}


class ReLU:
    def forward(self, x):
        self._cache = {'x': x}
        return np.maximum(0.0, x)

    def backward(self, dout):
        mask = self._cache['x'] > 0  # [N, D]
        return dout * mask

    def parameters(self):
        return {}

    def gradients(self):
        return {}


class Tanh:
    def forward(self, x):
        y = np.tanh(x)
        self._cache = {'y': y}
        return y

    def backward(self, dout):
        y = self._cache['y']
        return dout * (1.0 - y ** 2)

    def parameters(self):
        return {}

    def gradients(self):
        return {}


class Sigmoid:
    def forward(self, x):
        y = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        self._cache = {'y': y}
        return y

    def backward(self, dout):
        y = self._cache['y']
        return dout * y * (1.0 - y)

    def parameters(self):
        return {}

    def gradients(self):
        return {}


# ===================================================================
# Normalization
# ===================================================================

class LayerNorm:
    """Layer Normalization: normalize across feature dimension.

    For RL: invariant to batch composition, works with batch_size=1.
    y = gamma * (x - mu) / sqrt(sigma² + eps) + beta
    """

    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self._cache = {}

    def forward(self, x):
        """x: [N, D] → y: [N, D].  Normalize across dim=1 (feature axis)."""
        mu = x.mean(axis=1, keepdims=True)            # [N, 1]
        var = x.var(axis=1, keepdims=True)            # [N, 1]
        x_hat = (x - mu) / np.sqrt(var + self.eps)    # [N, D]
        self._cache['x_hat'] = x_hat
        self._cache['var'] = var
        self._cache['N'] = x.shape[0]
        return self.gamma * x_hat + self.beta         # [N, D]

    def backward(self, dout):
        """dout: [N, D] → dx: [N, D]"""
        x_hat = self._cache['x_hat']                  # [N, D]
        var = self._cache['var']                      # [N, 1]
        N = self._cache['N']
        D = x_hat.shape[1]

        dbeta = dout.sum(axis=0)                      # [D]
        dgamma = (dout * x_hat).sum(axis=0)           # [D]

        dx_hat = dout * self.gamma                    # [N, D]
        dvar = (dx_hat * x_hat * (-0.5) / (var + self.eps)).sum(axis=1, keepdims=True)  # [N, 1]
        dmu = dx_hat.sum(axis=1, keepdims=True) * (-1.0 / np.sqrt(var + self.eps)) + \
              dvar * (-2.0 * x_hat.mean(axis=1, keepdims=True))  # [N, 1]
        dx = dx_hat / np.sqrt(var + self.eps) + \
             dvar * 2.0 * x_hat / D + \
             dmu / D                                  # [N, D]

        self._cache['dgamma'] = dgamma
        self._cache['dbeta'] = dbeta
        return dx

    def parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def gradients(self):
        return {'gamma': self._cache.get('dgamma', 0),
                'beta': self._cache.get('dbeta', 0)}


class BatchNorm:
    """Batch Normalization: normalize across batch dimension.

    Cautions for RL: mixes independent trajectories, assumes large batch.
    For deployment (eval mode), use running mean/var.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
        self._cache = {}

    def forward(self, x):
        """x: [N, D] → y: [N, D].  Normalize across dim=0 (batch axis)."""
        if self.training:
            mu = x.mean(axis=0)                       # [D]
            var = x.var(axis=0)                       # [D]
            self.running_mean = self.momentum * self.running_mean + \
                                (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + \
                               (1 - self.momentum) * var
        else:
            mu = self.running_mean
            var = self.running_var

        x_hat = (x - mu) / np.sqrt(var + self.eps)    # [N, D]
        self._cache['x_hat'] = x_hat
        self._cache['var'] = var
        self._cache['N'] = x.shape[0]
        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        x_hat = self._cache['x_hat']                  # [N, D]
        var = self._cache['var']                      # [D]
        N = self._cache['N']

        dbeta = dout.sum(axis=0)                      # [D]
        dgamma = (dout * x_hat).sum(axis=0)           # [D]

        dx_hat = dout * self.gamma                    # [N, D]
        divar = (dx_hat * x_hat).sum(axis=0)          # [D]
        dxmu1 = dx_hat / np.sqrt(var + self.eps)      # [N, D]
        dsqrtvar = -divar / (var + self.eps)          # [D]
        dvar = 0.5 * dsqrtvar / np.sqrt(var + self.eps)  # [D]
        dsq = np.ones_like(x_hat) / N * dvar          # [N, D]
        dxmu2 = 2.0 * x_hat * dsq                     # [N, D]
        dx1 = dxmu1 + dxmu2                           # [N, D]
        dmu = -dx1.sum(axis=0)                        # [D]
        dx2 = np.ones_like(x_hat) / N * dmu           # [N, D]
        dx = dx1 + dx2                                # [N, D]

        self._cache['dgamma'] = dgamma
        self._cache['dbeta'] = dbeta
        return dx

    def parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def gradients(self):
        return {'gamma': self._cache.get('dgamma', 0),
                'beta': self._cache.get('dbeta', 0)}


# ===================================================================
# Dropout
# ===================================================================

class Dropout:
    """Dropout regularization — implicit ensemble during training.

    Training:  y = mask * x / (1-p)  (inverted dropout)
    Inference: y = x  (identity)

    Control-system pitfall: dropout at inference injects non-deterministic
    noise into action outputs — deployment MUST set training=False.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self._cache = {}

    def forward(self, x):
        if self.training and self.p > 0:
            mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            self._cache['mask'] = mask
            return x * mask
        self._cache['mask'] = np.ones_like(x)
        return x

    def backward(self, dout):
        return dout * self._cache['mask']

    def parameters(self):
        return {}

    def gradients(self):
        return {}


# ===================================================================
# Sequential container
# ===================================================================

class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.parameters().items():
                params[f'layer_{i}_{k}'] = v
        return params

    def gradients(self):
        grads = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.gradients().items():
                grads[f'layer_{i}_{k}'] = v
        return grads

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False


# ===================================================================
# Loss & Optimizer
# ===================================================================

class MSELoss:
    def forward(self, y_pred, y_true):
        self._cache = {'y_pred': y_pred, 'y_true': y_true}
        diff = y_pred - y_true  # [N, D]
        return float(np.mean(diff ** 2))

    def backward(self):
        y_pred = self._cache['y_pred']
        y_true = self._cache['y_true']
        N = y_pred.shape[0]
        return 2.0 * (y_pred - y_true) / N  # [N, D]


class CrossEntropyLoss:
    """Softmax + negative log-likelihood."""

    def forward(self, logits, y_true):
        """logits: [N, C], y_true: [N,] class indices."""
        N = logits.shape[0]
        shifted = logits - logits.max(axis=1, keepdims=True)     # stability
        exp_logits = np.exp(shifted)                              # [N, C]
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # [N, C]
        loss = -np.log(probs[np.arange(N), y_true] + 1e-12).mean()
        self._cache = {'probs': probs, 'y_true': y_true, 'N': N}
        return float(loss)

    def backward(self):
        probs = self._cache['probs']   # [N, C]
        y_true = self._cache['y_true']
        N = self._cache['N']
        grad = probs.copy()
        grad[np.arange(N), y_true] -= 1.0
        return grad / N                # [N, C]


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self._param_refs = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocities = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k in grads:
            g = grads[k] + self.weight_decay * self._param_refs[k]
            self._velocities[k] = self.momentum * self._velocities[k] - self.lr * g
            self._param_refs[k] += self._velocities[k]


# ===================================================================
# Gradient analysis utilities
# ===================================================================

def compute_gradient_stats(model, x_batch, loss_fn, y_true):
    """Forward-backward pass and return per-layer gradient statistics.

    Returns:
        dict: layer_name -> {'mean': float, 'std': float, 'norm': float,
                             'param_norm': float, 'grad_ratio': float}
    """
    y_pred = model.forward(x_batch)
    loss = loss_fn.forward(y_pred, y_true)
    model.backward(loss_fn.backward())

    stats = {}
    params = model.parameters()
    grads = model.gradients()
    for k in params:
        if k in grads:
            g = grads[k]
            p = params[k]
            param_norm = float(np.linalg.norm(p))
            grad_norm = float(np.linalg.norm(g))
            stats[k] = {
                'mean': float(np.mean(g)),
                'std': float(np.std(g)),
                'norm': grad_norm,
                'param_norm': param_norm,
                'grad_ratio': grad_norm / max(param_norm, 1e-10),
            }
    return stats, loss
