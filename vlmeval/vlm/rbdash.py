import os.path as osp
import warnings
from ..smp import *
from PIL import Image

import torch

from rbdash.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from rbdash.conversation import conv_templates, SeparatorStyle
from rbdash.model.builder import load_pretrained_model
from rbdash.utils import disable_torch_init
from rbdash.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


class RBDash:

    INSTALL_REQ = False

    def __init__(self, model_path="/path/to/rbdash-v1-13b", **kwargs):
        assert model_path is not None
        self.model_path = model_path

        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)

        # self.model.cuda()

        self.model_config = self.model.config

        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
    
    def get_image_and_text(self, question, image_path):
        qs = question
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def generate(self, image_path, prompt, dataset=None):
        input_ids, image_tensor = self.get_image_and_text(prompt, image_path)
        input_ids = input_ids.unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)

        stop_str = conv_templates['vicuna_v1'].sep if conv_templates['vicuna_v1'].sep_style != SeparatorStyle.TWO else conv_templates['vicuna_v1'].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs
