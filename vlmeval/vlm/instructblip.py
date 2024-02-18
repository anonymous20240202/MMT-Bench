# import torch
# from PIL import Image
# from abc import abstractproperty
# import os.path as osp
# import os 
# from ..smp import *


# class InstructBLIP:

#     INSTALL_REQ = True

#     def __init__(self, name):
#         self.config_map = {
#             'instructblip_7b': f'misc/blip2_instruct_vicuna7b.yaml', 
#             'instructblip_13b': f'misc/blip2_instruct_vicuna13b.yaml', 
#         }

#         self.file_path = __file__
#         config_root = osp.dirname(self.file_path)
            
#         try:
#             from lavis.models import load_preprocess
#             from omegaconf import OmegaConf
#             from lavis.common.registry import registry
#         except:
#             warnings.warn("Please install lavis before using InstructBLIP. ")
#             exit(-1)

#         assert name in self.config_map
#         cfg_path = osp.join(config_root, self.config_map[name])
#         cfg = OmegaConf.load(cfg_path)

#         model_cfg = cfg.model
#         assert osp.exists(model_cfg.llm_model) or splitlen(model_cfg.llm_model) == 2
#         model_cls = registry.get_model_class(name="blip2_vicuna_instruct")
#         model = model_cls.from_config(model_cfg)
#         model.eval()

#         self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#         device = self.device
#         model.to(device)
#         self.model = model
#         self.kwargs = {'max_length': 128}

#         preprocess_cfg = cfg.preprocess
#         vis_processors, _ = load_preprocess(preprocess_cfg)
#         self.vis_processors = vis_processors

#     def generate(self, image_path, prompt, dataset=None):
#         vis_processors = self.vis_processors
#         raw_image = Image.open(image_path).convert('RGB')
#         image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
#         outputs = self.model.generate(dict(image=image_tensor, prompt=prompt))
#         return outputs[0]

import warnings
from ..smp import *
from PIL import Image
import torch

# pip install accelerate
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


class InstructBLIP:
    INSTALL_REQ = False

    def __init__(self, model_path="Salesforce/instructblip-vicuna-7b", **kwargs):
        assert model_path is not None
        self.model_path = model_path

        self.processor = InstructBlipProcessor.from_pretrained(model_path)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_path).cuda()

        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def generate(self, image_path, prompt, dataset=None):
        raw_image = Image.open(image_path).convert('RGB')

        inputs = self.processor(raw_image, prompt, return_tensors="pt").to("cuda")
        try:
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=1024,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1.,
            )
            # except:
            #     print(f"================={image_path} {prompt}")
            output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        except:
            output = "The input text is too long."
        
        return output
