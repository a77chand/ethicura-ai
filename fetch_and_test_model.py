import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
from huggingface_hub import snapshot_download

print("Downloading repo snapshot (first run only, ~cached after)...")
local_dir = snapshot_download(repo_id="ariyul/deepfake-detection-cnn_v2")

model_path = os.path.join(local_dir, "keras_model", "best_model.keras")
print(f"\nLoading model from: {model_path}")
model = keras.saving.load_model(model_path)
model.summary()

print("\n--- Layer names ---")
for layer in model.layers:
    print(f"  {layer.name:40s}  {layer.__class__.__name__}")

print("\n--- Test prediction on random noise ---")
dummy_input = np.random.rand(1, 224, 224, 3).astype("float32")
pred = model.predict(dummy_input, verbose=0)
print(f"Output shape: {pred.shape}, value: {pred}")
