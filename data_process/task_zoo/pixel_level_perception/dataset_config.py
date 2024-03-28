dataset = {
    # ==============================================================================================================================
    # low_level_vision
    "depth_estimation": {
        "dataset_list": [
            "taskonomy",
            "nyu",
            "nuscenes",
            "kitti"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/low_level_vision/shape_recognition"
    },
    "height_estimation": {
        "dataset_list": [
            "gta_height",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/low_level_vision/height_depth_estimation"
    },
    # ==============================================================================================================================
    # visual_recognition
    "age_gender_race_recognition": {
        "dataset_list": [
            "FairFace",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/age_gender_race_recognition"
    },
    "animated_character_recognition": {
        "dataset_list": [
            "Anime_Characters_Personality_And_Facial_Images",
            "moeimouto_faces"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/animated_character_recognition"
    },
    "celebrity_recognition": {
        "dataset_list": [
            "lfw",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/celebrity_recognition"
    },
    "weapon_recognition": {
        "dataset_list": [
            "OWAD",
            "weapon_detection_dataset"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/weapon_recognition"
    },
    "building_recognition": {
        "dataset_list": [
            "ArchitecturalStyles",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/building_recognition"
    },
    "deepfake_detection": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/deepfake_detection/metadata_info.json",
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/deepfake_detection",
        "sampling_num": {
            "FFplusplus": 100,
            "CelebDFv2": 50,
            "dalle_art_deepfake": 50,
        }
    },
    "rock_recognition": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/rock_visual_recognition/metadata_info.json",
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/rock_recognition",
        "sampling_num": {
            "rock_image_recognition1": 100,
            "rock_image_recognition2": 100
        }
    },
    "disaster_recognition": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/disaster_visual_recognition/metadata_info.json",
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/disaster_recognition",
        "sampling_num": {
            "disaster_image_recognition": 100,
            "MEDIC": 100
        }
    },
    "weather_recognition": {
        "dataset_list": [
            "weather_image_recognition",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/weather_recognition"
    },
    "gesture_recognition": {
        "dataset_list": [
            "CNNGestureRecognizer",
            "gesture_digits"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/gesture_recognition"
    },
    "profession_recognition": {
        "dataset_list": [
            "IdenProf",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/profession_recognition"
    },
    "electronic_object_recognition": {
        "dataset_list": [
            "e_waste",
            "electronics_object_image_dataset_computer_parts"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/electronic_object_recognition"
    },
    "plant_recognition": {
        "dataset_list": [
            "flower_photos",
            "Plant_Data"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/plant_recognition"
    },
    "vehicle_recognition": {
        "dataset_list": [
            "TAU_Vehicle_Type_Recognition",
            "vehicle_type_recognition"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/vehicle_recognition"
    },
    "shape_recognition": {
        "dataset_list": [
            "twod_geometric_shapes_dataset",
            "gpt_auto_generated_shape"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/shape_recognition"
    },
    "color_recognition": {
        "dataset_list": [
            "python_auto_generated_color_name",
            "python_auto_generated_color_rgb"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition"
    },
    "texture_material_recognition": {
        "dataset_list": [
            "kth",
            "kyberge",
            "uiuc",
            "opensurfaces"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/texture_material_recognition"
    },
    "painting_recognition": {
        "dataset_list": [
            "wikiart",
            "best_artwork_of_all_time",
            "van_gogh_paintings_dataset",
            "chinese_patinting_internet"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/painting_recognition"
    },
    "sculpture_recognition": {
        "dataset_list": [
            "sculpture_internet",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/sculpture_recognition"
    },
    "logo_and_brand_recognition": {
        "dataset_list": [
            "fake_real_logo_detection_dataset",
            "flickr_sport_logos_10",
            # "car_logos_dataset",
            # "logo627"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/logo_and_brand_recognition"
    },
    "image_season_recognition": {
        "dataset_list": [
            "image_season_recognition"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/image_season_recognition"
    },
    "astronomical_recognition": {
        "dataset_list": [
            "astronomical_internet"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/astronomical_recognition"
    },
    "sports_recognition": {
        "dataset_list": [
            "Sports_Image_Classification_100",
            "Cricket_Football_Baseball"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/sports_recognition"
    },
    "food_recognition": {
        "dataset_list": [
            "Fruits_and_Vegetables",
            "food_101"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/food_recognition"
    },
    "waste_recognition": {
        "dataset_list": [
            "Garbage_Classification_12",
            "Waste_Classification_data"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/waste_recognition"
    },
    "animals_recognition": {
        "dataset_list": [
            "animals90",
            "animals150",
            "animals10"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/animals_recognition"
    },
    "muscial_instrument_recognition": {
        "dataset_list": [
            "musical_instruments_image_classification",
            "music_instruments_classification"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/muscial_instrument_recognition"
    },
    "film_and_television_recognition": {
        "dataset_list": [
            "internet_poster",
            "movie_posters_kaggle"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/film_and_television_recognition"
    },
    "chemical_apparatusn_recognition": {
        "dataset_list": [
            "chemical_apparatus_image_dataset",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/chemical_apparatusn_recognition"
    },
    "religious_recognition": {
        "dataset_list": [
            "religious_symbols_image_classification",
            "dataset_of_traditional_chinese_god_statue"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/religious_recognition"
    },
    "scene_recognition": {
        "dataset_list": [
            "indoor_scene_recognition",
            "places365"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/scene_recognition"
    },
    "landmark_recognition": {
        "dataset_list": [
            "landmark_internet",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/landmark_recognition"
    },
    "national_flag_recognition": {
        "dataset_list": [
            "country_flag"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/national_flag_recognition"
    },
    "fashion_recognition": {
        "dataset_list": [
            "fashion_mnist",
            "deepfashion"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/fashion_recognition"
    },
    "abstract_visual_recognition": {
        "dataset_list": [
            "quickdraw",
            "imagenet_sketch"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/abstract_visual_recognition"
    },
    # ==============================================================================================================================
    # localization
    "face_detection": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/face_detection/metadata_info.json",
        "sampling_num": {
            "FDDB": 100,
            "WIDERFACE": 100
        },
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/face_detection"
    },
    "salient_object_detection_rgb": {
        "dataset_list": [
            "MSRA10K",
            "DUTS",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/salient_object_detection_rgb"
    },
    "salient_object_detection_rgbd": {
        "dataset_list": [
            "DES",
            "NJU2K",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/salient_object_detection_rgbd"
    },
    "transparent_object_detection": {
        "dataset_list": [
            "Trans10K",
            "Transparent_Object_Images",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/transparent_object_detection"
    },
    "camouflage_object_detection": {
        "dataset_list": [
            "COD10K",
            "NC4K",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/camouflage_object_detection"
    },
    "remote_sensing_object_detection": {
        "dataset_list": [
            "DIOR",
            "VisDrone",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/remote_sensing_object_detection"
    },
    "object_detection": {
        "dataset_list": [
            "coco_det",
            "VOC2012_det",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/object_detection"
    },
    "small_object_detection": {
        "dataset_list": [
            "sod4bird",
            "drone2021",
            "tinyperson"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/small_object_detection"
    },
    "rotated_object_detection": {
        "dataset_list": [
            "dota",
            "ssdd_inshore",
            "ssdd_offshore"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/localization/rotated_object_detection"
    },
    # ==============================================================================================================================
    # pixel-level perception
    "pixel_localization": {
        "dataset_list": [
            "coco_pixel_localization",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/pixel_level_perception/pixel_localization"
    },
    "polygon_localization": {
        "dataset_list": [
            "coco_polygon",
            # "youtubevis2019_polygon",
            # "ovis_polygon"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/pixel_level_perception/polygon_localization"
    },
    "image_matting": {
        "dataset_list": [
            "am2k",
            "aim500"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/pixel_level_perception/image_matting"
    },
    "pixel_recognition": {
        "dataset_list": [
            "coco_pixel_recognition",
            "ade20k_pixel_recognition"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/pixel_level_perception/pixel_recognition"
    },
    "interactive_segmentation": {
        "dataset_list": [
            "davis_interactive",
            "berkley_interactive"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/pixel_level_perception/interactive_segmentation"
    },
    # ==============================================================================================================================
    # ocr
    "handwritten_text_recognition": {
        "dataset_list": [
            "iam_line",
            "iam_page"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/ocr/handwritten_text_recognition"
    },
    "handwritten_mathematical_expression_recognition": {
        "dataset_list": [
            "hme100k",
            "crohme2014"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/ocr/handwritten_mathematical_expression_recognition"
    },
    "font_recognition": {
        "dataset_list": [
            "adobe_vfr",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/ocr/font_recognition"
    },
    "scene_text_recognition": {
        "dataset_list": [
            "ICDAR2013",
            "IIIT5K"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/ocr/scene_text_recognition"
    },
    # ==============================================================================================================================
    # visual_prompt_understanding
    "visual_prompt_understanding": {
        "dataset_list": [
            "vipbench"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_prompt_understanding/visual_prompt_understanding"
    },
    "som_recognition": {
        "dataset_list": [
            "sombench_flickr30k_grounding",
            "sombench_refcocog_refseg"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_prompt_understanding/som_recognition"
    },
    # ==============================================================================================================================
    # image2image_translation
    "jigsaw_puzzle_solving": {
        "dataset_list": [
            "jigsaw_puzzle_solving_natural",
            "jigsaw_puzzle_solving_painting"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving"
    },
    # emotion
    "facial_expression_recognition": {
        "dataset_list": [
            "ferplus",
            "facial_emotion_recognition_dataset"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/facial_expression_recognition"
    },
    "micro_expression_recognition": {
        "dataset_list": [
            "CASME",
            "SAMM"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/micro_expression_recognition"
    },
    "body_emotion_recognition": {
        "dataset_list": [
            # "emotic",
            "CAERS",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/body_emotion_recognition"
    },
    "artwork_emotion_recognition": {
        "dataset_list": [
            "artemis",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/artwork_emotion_recognition"
    },
    "scene_emotion_recognition": {
        "dataset_list": [
            "Artphoto",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/scene_emotion_recognition"
    },
    "facail_expression_change_recognition": {
        "dataset_list": [
            "emotion_change",
            "ferg_db"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/facail_expression_change_recognition"
    },
    # ==============================================================================================================================
    # visual_grounding
    "referring_detection": {
        "dataset_list": [
            "RefCOCO_refer",
            "RefCOCOplus_refer",
            "RefCOCOg_refer"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_grounding/referring_detection"
    },
    "reason_seg": {
        "dataset_list": [
            "reason_seg",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_grounding/reason_seg"
    },
    # ==============================================================================================================================
    # relation reasoning
    "scene_graph_recognition": {
        "dataset_list": [
            # "visual_genome_sg",
            # "vrd_sg",
            "vsr"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/scene_graph_recognition",
    },
    "social_relation_recognition": {
        "dataset_list": [
            "social_relation_dataset",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/social_relation_recognition",
    },
    "human_object_interaction_recognition": {
        "dataset_list": [
            "hicodet",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human_object_interaction_recognition",
    },
    "human_interaction_understanding": {
        "dataset_list": [
            "hicodet_hiu",
            "bit"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human_interaction_understanding",
    },
    # ==============================================================================================================================
    # visual_captioning
    "writing_poetry_from_image": {
        "dataset_list": [
            "poetry"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_captioning/writing_poetry_from_image",
    },
    "image_captioning": {
        "dataset_list": [
            "coco_captions"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_captioning/image_captioning",
    },
    "instance_captioning": {
        "dataset_list": [
            "visual_genome_caption",
            "refcocog_caption"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_captioning/instance_captioning",
    },
    "multiple_instance_captioning": {
        "dataset_list": [
            # "arch",
            "flickr30k_multi_regions"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_captioning/multiple_instance_captioning",
    },
    "image_captioning_paragraph": {
        "dataset_list": [
            # "ade20k_caption",
            "paragraph_caption"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_captioning/image_captioning_paragraph",
    },
    # ==============================================================================================================================
    # visual_illusion
    "color_assimilation": {
        "dataset_list": [
            "gvil_assimilation"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/color_assimilation",
    },
    "color_constancy": {
        "dataset_list": [
            "gvil_constancy"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/color_constancy",
    },
    "color_contrast": {
        "dataset_list": [
            "gvil_contrast"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/color_contrast",
    },
    "geometrical_perspective": {
        "dataset_list": [
            "gvil_perspective"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/geometrical_perspective",
    },
    "geometrical_relativity": {
        "dataset_list": [
            "gvil_relativity"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/geometrical_relativity",
    },
    # ==============================================================================================================================
    # visual_coding
    "eqn2latex": {
        "dataset_list": [
            "im2latex90k"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_code/eqn2latex",
    },
    "screenshot2code": {
        "dataset_list": [
            "pix2code_andriod",
            "pix2code_ios",
            "pix2code_web"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_code/screenshot2code",
    },
    "sketch2code": {
        "dataset_list": [
            "sketch2code_kaggle"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_code/sketch2code",
    },
    # ==============================================================================================================================
    # counting
    "crowd_counting": {
        "dataset_list": [
            "ShanghaiTech",
            "CARPK"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/counting/crowd_counting",
    },
    "counting_by_category": {
        "dataset_list": [
            "fsc147_category",
            "countqa_vqa",
            "countqa_cocoqa",
            "tallyqa_simple"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/counting/counting_by_category",
    },
    "counting_by_reasoning": {
        "dataset_list": [
            "tallyqa_complex",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/counting/counting_by_reasoning",
    },
    "counting_by_visual_prompting": {
        "dataset_list": [
            "fsc147"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/counting/counting_by_visual_prompting",
    },
    # keypoint_detection
    "human_keypoint_detection": {
        "dataset_list": [
            "MSCOCO_keypoint",
            "MPII_keypoint"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/keypoint_detection/human_keypoint_detection",
    },
    "furniture_keypoint_detection": {
        "dataset_list": [
            "furniture_keypoint",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/keypoint_detection/furniture_keypoint_detection",
    },
    "animal_keypoint_detection": {
        "dataset_list": [
            "ap10k_keypoint",
            "Animal_kingdom_keypoint"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/keypoint_detection/animal_keypoint_detection",
    },
    "clothes_keypoint_detection": {
        "dataset_list": [
            "Deepfashion_keypoint",
            "Deepfashion2_keypoint"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/keypoint_detection/clothes_keypoint_detection",
    },
    "vehicle_keypoint_detection": {
        "dataset_list": [
            "vehicle_keypoint",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/keypoint_detection/vehicle_keypoint_detection",
    },
    # action recognition
    "image_based_action_recognition": {
        "dataset_list": [
            "HAR",
            "POLAR"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/image_based_action_recognition",
    },
    "sign_language_recognition": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/sign_language_recognition/metadata_info.json",
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/sign_language_recognition",
        "sampling_num": {
            "MSASL": 100,
            "WASAL": 100
        }
    },
    "general_action_recognition": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/general_action_recognition/metadata_info.json",
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/general_action_recognition",
        "sampling_num": {
            "kinetics400": 200
        }
    },
    "action_quality_assessment": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/action_quality_assessment/metadata_info.json",
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/action_quality_assessment",
        "sampling_num": {
            "UNLV": 100,
            "AQA7": 100
        }
    },
    # ==============================================================================================================================
    # doc_understanding
    "visual_document_information_extraction": {
        "dataset_list": [
            "sroie",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/doc_understanding/visual_document_information_extraction",
    },
    "table_structure_recognition": {
        "dataset_list": [
            "scitsr",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/doc_understanding/table_structure_recognition",
    },
    # ==============================================================================================================================
    # intelligence_quotient_test
    "ravens_progressive_matrices": {
        "metadata_info": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/intelligence_quotient_test/ravens_progressive_matrices/metadata_info.json",
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/intelligence_quotient_test/ravens_progressive_matrices",
        "sampling_num": {
            "RAVEN_10000": 200,
        }
    },
    # ==============================================================================================================================
    # cross_image_tracking
    "single_object_tracking": {
        "dataset_list": [
            "youtubevis2019_sot",
            "ovis_sot",
            # "vot2018"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/cross_image_matching/single_object_tracking",
    },
    "point_tracking": {
        "dataset_list": [
            "tapvid_davis",
            "tapvid_rgb_stacking",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/cross_image_matching/point_tracking",
    },
    "one_shot_detection": {
        "dataset_list": [
            "fss_1000",
            "paco_part"
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/cross_image_matching/one_shot_detection",
    },
    # ==============================================================================================================================
    # image_evaluation_judgement
    "lvlm_response_judgement": {
        "dataset_list": [
            "LVLM_eHub_conv_data",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/image_evaluation_judgement/lvlm_response_judgement",
    },
    # ==============================================================================================================================
    # medical_understanding
    "medical_modality_recognition": {
        "dataset_list": [
            "medical_modality_recognition",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/medical_understanding/medical_modality_recognition",
    },
    "anatomy_identification": {
        "dataset_list": [
            "anatomy_identification",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/medical_understanding/anatomy_identification",
    },
    "disease_diagnose": {
        "dataset_list": [
            "disease_diagnose",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/medical_understanding/disease_diagnose",
    },
    "lesion_grading": {
        "dataset_list": [
            "lesion_grading",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/medical_understanding/lesion_grading",
    },
    "other_biological_attributes": {
        "dataset_list": [
            "other_biological_attributes",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/medical_understanding/other_biological_attributes",
    },
    # ==============================================================================================================================
    # gui_navigation
    "gui_general": {
        "dataset_list": [
            "gui_general",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/gui_navigation/gui_general",
    },
    "google_apps": {
        "dataset_list": [
            "google_apps",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/gui_navigation/google_apps",
    },
    "gui_install": {
        "dataset_list": [
            "google_apps",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/gui_navigation/gui_install",
    },
    "web_shopping": {
        "dataset_list": [
            "google_apps",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/gui_navigation/web_shopping",
    },
    # ==============================================================================================================================
    # image_retrieval
    "person_reid": {
        "dataset_list": [
            "market_1501",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/image_retrieval/person_reid",
    },
    "image2image_retrieval": {
        "dataset_list": [
            "places365_retrieval",
        ],
        "output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/image_retrieval/image2image_retrieval",
    },
}
