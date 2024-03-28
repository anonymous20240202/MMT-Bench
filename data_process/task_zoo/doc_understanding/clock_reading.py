from pathlib import Path

from base_dataset import BaseDataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from tqdm import tqdm
import os

def draw_clock(hour, minute, save_path):
    # Convert the hour to 12-hour format and calculate angles for the hands
    hour = hour % 12
    hour_angle = (hour + minute / 60) * 30  # 360 degrees / 12 hours
    minute_angle = minute * 6  # 360 degrees / 60 minutes

    # Create a new figure
    fig, ax = plt.subplots()

    # Draw the clock face
    clock_face = plt.Circle((0, 0), 1, color='white', ec='black')
    ax.add_patch(clock_face)

    angel_dict = {}
    # Draw the clock numbers
    for i in range(1, 13):
        angle = np.radians(- i * 30 + 90)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.text(x * 0.85, y * 0.85, str(i), horizontalalignment='center', verticalalignment='center')

    # Draw the hour hand
    hour_angle = np.radians(-hour_angle + 90)
    ax.add_patch(patches.FancyArrow(0, 0, 0.5 * np.cos(hour_angle), 0.5 * np.sin(hour_angle), 
                                    width=0.05, head_width=0.1, head_length=0.1, color='black'))

    # Draw the minute hand
    minute_angle = np.radians(-minute_angle + 90)
    ax.add_patch(patches.FancyArrow(0, 0, 0.7 * np.cos(minute_angle), 0.7 * np.sin(minute_angle), 
                                    width=0.02, head_width=0.05, head_length=0.1, color='blue'))

    # Set up the plot limits and aspect
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', 'box')
    ax.axis('off')  # Hide the axes

    # Show the plot
    # plt.show()
    plt.savefig(save_path,  bbox_inches='tight', dpi= 150)


class clock_reading(BaseDataset):
    DATA_METAINFO = {
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/doc_understanding/clock_reading/images",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()

        for i in tqdm(range(400)):
            hour = random.randint(0, 11)
            minute = random.randint(0, 59)
            
            save_image_path = os.path.join(self.save_image_path, BaseDataset.new_image_name())

            draw_clock(hour, minute, save_image_path)

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": save_image_path,
                    "hour": hour,
                    "minute": minute
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random

        num_choices = 4
        question = f"What is the time of the clock in the picture? The thick line represents the clock line and the blue line represents the minute line."

        gt = f"{image_info['hour']} : {image_info['minute']}"

        i = 0
        while i <= 10:
            try:
                wrong_choice_list = []
                for i in range(num_choices - 1):
                    hour = random.randint(0, 11)
                    minute = random.randint(0, 59)
                    wrong_choice_list.append(f"{hour} : {minute}")
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": wrong_choice_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
        