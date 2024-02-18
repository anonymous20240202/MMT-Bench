import os.path as osp
import warnings
from ..smp import *
from PIL import Image
import torch

from SPHINX import SPHINXModel
from PIL import Image
import torch
import torch.distributed as dist


class Sphinx:
    INSTALL_REQ = False

    def __init__(self, model_path="/path/to/SPHINX_v2-1k", **kwargs):
        assert model_path is not None
        self.model_path = model_path

        # rank, world_size = get_rank_and_world_size()
        # torch.cuda.set_device(rank)

        # self.model = SPHINXModel.from_pretrained(
        #     pretrained_path=model_path, with_visual=True,
        #     mp_group=dist.new_group(ranks=list(range(world_size))))

        self.model = SPHINXModel.from_pretrained(pretrained_path=model_path, with_visual=True)

    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path)
        qas = [[prompt, None]]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            response = self.model.generate_response(qas, image, max_gen_len=16, temperature=0.9, top_p=0.5, seed=0)

        return response
