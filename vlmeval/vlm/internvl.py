import torch
from PIL import Image
from abc import abstractproperty
import os
import os.path as osp
from ..smp import *
from ..utils import DATASET_TYPE, CustomPrompt



import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer


class InternVL(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self, 
                 name,
                 path="OpenGVLab/InternVL-Chat-Chinese-V1-1",
                 **kwargs): 

        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto').eval()
        self.model_path_map = path

        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.image_processor = CLIPImageProcessor.from_pretrained(path)



        # assert name in self.model_path_map or osp.exists(name) or splitlen(name) == 2
        # if name in self.model_path_map:
        #     model_path = self.model_path_map[name]
        # else:
        #     model_path = name

        # assert osp.exists(model_path) or splitlen(model_path) == 2
        
        # model_name = 'llava-v1.5-7b' if model_path == 'Lin-Chen/ShareGPT4V-7B' else get_model_name_from_path(model_path)
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        #     model_path=model_path, 
        #     model_base=None, 
        #     model_name=model_name, 
        #     device='cpu', 
        #     device_map='cpu'
        # )
        # self.model = self.model.cuda()
        # self.conv_mode =  'llava_v1'

        kwargs_default = dict(do_sample=False, temperature=0.2, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False
    
    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += "\n请直接回答选项字母。" if cn_string(prompt) else "\nAnswer with the option's letter from the given choices directly. Note that one and only one of the above options is correct."
        else:
            prompt += "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."

        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):

        image = Image.open(image_path).convert('RGB')
        image = image.resize((448, 448))
        pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )
        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)

        return response


        # args = abstractproperty()
        # args.image_aspect_ratio = 'pad'
        # image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        # if self.model.config.mm_use_im_start_end:
        #     inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        # else:
        #     inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        # conv = conv_templates[self.conv_mode].copy()
        # conv.append_message(conv.roles[0], inp)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # with torch.inference_mode():
        #     output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        # output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]
        # return output
