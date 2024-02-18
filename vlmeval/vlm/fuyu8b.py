import os.path as osp
import warnings
from ..smp import *
from PIL import Image


class Fuyu8B:

    INSTALL_REQ = False

    def __init__(self, model_path="adept/fuyu-8b", **kwargs):
        assert model_path is not None
        self.model_path = model_path
        
        from transformers import FuyuProcessor, FuyuForCausalLM

        self.processor = FuyuProcessor.from_pretrained(model_path)
        self.model = FuyuForCausalLM.from_pretrained(model_path, device_map="cuda")

        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")

        generation_output = self.model.generate(**inputs, max_new_tokens=64, pad_token_id=self.processor.tokenizer.eos_token_id)
        output = self.processor.batch_decode(generation_output[:, -64:], skip_special_tokens=True)[0]
        
        return output
