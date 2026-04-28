import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from land_change_detection.remote_sensing_vlm import RemoteSensingQwen2VL2B

model_path = "akshaydudhane/EarthDial_4B_RGB"
vlm = RemoteSensingQwen2VL2B(model_name_or_path=model_path, device="mps")
print("Backend initialized: ", vlm.backend)

before = np.zeros((224, 224, 3), dtype=np.uint8)
after = np.zeros((224, 224, 3), dtype=np.uint8)
overlay = np.zeros((224, 224, 3), dtype=np.uint8)

res = vlm.explain(before, after, overlay)
print("RAW TEXT:")
print(res.raw_text)
print("PARSED:")
print(res.parsed)
