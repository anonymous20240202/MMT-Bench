import warnings
from ..smp import *
from PIL import Image
import torch

# pip install accelerate
import torch
from PIL import Image

import cv2
import torch
import sys
from PIL import Image
sys.path.append('/path/to/model_zoo/LLaMA-Adapter/llama_adapter_v2_multimodal7b')
# import llama


class llama_adapter:
    INSTALL_REQ = False

    def __init__(self, model_path="/path/to/llama_model_weights", **kwargs):

        assert model_path is not None
        self.model_path = model_path

        # choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
        self.model, self.preprocess = llama.load("LORA-BIAS-7B-v21", model_path, llama_type="7B", device="cuda")
        self.model.eval()

        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def generate(self, image_path, prompt, dataset=None):
        prompt = llama.format_prompt(prompt)
        img = Image.fromarray(cv2.imread(image_path))
        img = self.preprocess(img).unsqueeze(0).to("cuda")

        output = self.model.generate(img, [prompt])[0]

        return output
