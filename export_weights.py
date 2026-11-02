"""Export the trained model's weights (models/mnist_cnn.keras or .h5) to docs/weights.bin for the browser demo.

Run from the project root, inside your normal virtual environment:
    python3 export_weights.py
"""
import os
import sys

import numpy as np
import tensorflow as tf

if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
elif os.path.exists('models/mnist_cnn.keras'):
    MODEL_PATH = 'models/mnist_cnn.keras'   # new format
else:
    MODEL_PATH = 'models/mnist_cnn.h5'      # old format
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else 'docs/weights.bin'
EXPECTED_PARAMS = 225034

model = tf.keras.models.load_model(MODEL_PATH)

parts = []
for layer in model.layers:
    for w in layer.get_weights():  # kernel, then bias
        parts.append(w.astype('<f4').ravel())

flat = np.concatenate(parts)
if flat.size != EXPECTED_PARAMS:
    raise SystemExit(f'Expected {EXPECTED_PARAMS} values but found {flat.size}. '
                     'Did the model architecture change?')

os.makedirs(os.path.dirname(OUT_PATH) or '.', exist_ok=True)
flat.tofile(OUT_PATH)
print(f'Wrote {flat.size:,} values ({flat.nbytes / 1024:.0f} KB) to {OUT_PATH}')