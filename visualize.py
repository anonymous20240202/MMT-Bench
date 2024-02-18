import os
from collections import defaultdict

import gradio as gr
import pandas as pd
from tqdm import tqdm

from vlmeval.smp.vlm import decode_base64_to_image_file


def exist_or_mkdir(folder_path):
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)


TEMP_VISUALIZE_PATH = "visual_image_path"
exist_or_mkdir(TEMP_VISUALIZE_PATH)


def get_data_list():

    file_path = "./LMUData/mmtbench.tsv"
    df = pd.read_csv(file_path, sep='\t')

    data_list = defaultdict(list)

    for data_line in tqdm(df.iloc):
        index = data_line['index']
        question = data_line["question"]
        answer = data_line["answer"]
        A, B, C, D, E, F, G, H, I =  data_line[3: 12]

        image = data_line["image"]
        category = data_line["category"]

        data_list[category].append(
            {
                "image": image,
                'index': index,
                "question": question,
                "answer": answer,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "E": E,
                "F": F,
                "G": G,
                "H": H,
                "I": I
            }
        )

    print(len(data_list.keys()))
    return data_list
        

data_list = get_data_list()


current_category = list(data_list.keys())[0]
current_index = 0


def display_data(category, index):
    global current_category, current_index
    current_category = category
    current_index = index

    category_items = data_list.get(category, [])
    if index < 0 or index >= len(category_items):
        return "Invalid index for the selected category.", None, None

    data = category_items[index]

    options_string = f"Options\n"
    if data.get("A", None) and str(data.get("A", None)) != 'nan':
        options_string += f"(A). {data['A']}\n"
    if data.get("B", None) and str(data.get("B", None)) != 'nan':
        options_string += f"(B). {data['B']}\n"
    if data.get("C", None) and str(data.get("C", None)) != 'nan':
        options_string += f"(C). {data['C']}\n"
    if data.get("D", None) and str(data.get("D", None)) != 'nan':
        options_string += f"(D). {data['D']}\n"
    if data.get("E", None) and str(data.get("E", None)) != 'nan':
        options_string += f"(E). {data['E']}\n"
    if data.get("F", None) and str(data.get("F", None)) != 'nan':
        options_string += f"(F). {data['F']}\n"
    if data.get("G", None) and str(data.get("G", None)) != 'nan':
        options_string += f"(G). {data['G']}\n"
    if data.get("H", None) and str(data.get("H", None)) != 'nan':
        options_string += f"(H). {data['H']}\n"
    if data.get("I", None) and str(data.get("I", None)) != 'nan':
        options_string += f"(I). {data['I']}\n"
    
    spilt_line = "=========================================================="

    show_string = f"Question: {data['question']}\n{spilt_line}\n{options_string}{spilt_line}\nAnswer: {data['answer']}"

    image_file_path = f"{TEMP_VISUALIZE_PATH}/{data['index']}.jpg"
    decode_base64_to_image_file(data['image'], image_file_path)
    return show_string, image_file_path, f'{index+1}/{len(data_list.get(category, []))}'


def next_item():
    global current_index
    category_items = data_list.get(current_category, [])
    current_index = min(current_index + 1, len(category_items) - 1)
    return display_data(current_category, current_index)


def previous_item():
    global current_index
    category_items = data_list.get(current_category, [])
    current_index = max(current_index - 1, 0)
    return display_data(current_category, current_index)


with gr.Blocks() as iface:
    with gr.Row():
        category_dropdown = gr.Dropdown(list(data_list.keys()), label="Select Category")
        prev_button = gr.Button("Previous")
        next_button = gr.Button("Next")
    index_output = gr.Textbox(label="Current Index", interactive=False)
    question_output = gr.Textbox(label="Question and Options")
    image_output = gr.Image(label="Image", width=768)
    # index_output = gr.Textbox(label="Current Index", interactive=False)

    category_dropdown.change(
        lambda category: display_data(category, 0), 
        inputs=[category_dropdown], 
        outputs=[question_output, image_output, index_output]
    )
    prev_button.click(
        previous_item, [], [question_output, image_output, index_output]
    )
    next_button.click(
        next_item, [], [question_output, image_output, index_output]
    )

iface.queue(api_open=True).launch(share=True)
