# -*- coding: utf-8 -*-
import faulthandler
# ??import???????????????????????
faulthandler.enable()
# ???????��??????
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
import glob
import cv2
import json
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
from torchvision import transforms
import numpy as np
from ResNet_18_34 import ResNet18
import shutil


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator
)

import matplotlib.pyplot as plt
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.3):
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))
    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def extract_data_from_json(data, key_path):
    # Traverse the JSON structure using the key path
    current_data = data
    try:
        for key in key_path:
            current_data = current_data[key]
        return current_data
    except (KeyError, TypeError):
        return None

def process_image(
        model_config="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights1="weights/groundingdino_swint_ogc.pth",
        model_weights2="weights/model_weights840.pth",
        model_weights3="weights/model_weights200.pth",
        image_path="D:/postgraduate/Grounding-Dino-FineTuning-main/.asset/test1.jpg",
        text_prompt1="leaf",
        text_prompt2="spot lesions",
        box_threshold=0.45,
        text_threshold=0.35,
        image_output_dir = "./outputs"
):
    ##????
    # ???????
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.416, 0.468, 0.355], [0.210, 0.206, 0.213])])

    labels_plant = ['Corn', 'Other', 'Potato', 'Rice', 'Sorghum', 'Soybean', 'Sweet_Potato', 'Wheat']

    labels_disease = [
        ['corn bacterial red stripe', 'corn bacterial stripe', 'corn bacterial wilt', 'southern corn leaf blight',
         'corn brown spot', 'corn common rust', 'corn curvularia leaf spot', 'corn genetic streak',
         'corn gray leaf spot', 'corn healthy', 'corn leaf spot', 'northern corn leaf blight', 'corn round spot',
         'corn rust', 'corn sheath blight', 'corn sooty blotch', 'corn streak virus', 'corn virus'],
        ['potato early blight', 'potato healthy', 'potato late blight'],
        ['rice bacterial blight', 'rice brown spot', 'rice healthy', 'rice leaf blast', 'rice false smut',
         'rice tungro disease'],
        ['sorghum anthracnose', 'sorghum bacterial red stripe', 'sorghum bacterial stripe', 'sorghum dwarf mosaic',
         'sorghum gloeocercospora leaf spot', 'sorghum glume mold', 'sorghum mosaic',
         'sorghum mycosphaerella leaf spot', 'sorghum northern leaf blight', 'sorghum sheath blight',
         'sorghum target spot'],
        ['soybean alternaria black spot', 'soybean anthracnose', 'soybean bacterial spot', 'soybean blight',
         'soybean downy mildew', 'soybean gray mold', 'soybean gray spot', 'soybean healthy', 'soybean leaf blight',
         'soybean mosaic', 'soybean phytotoxicity', 'soybean rust', 'soybean sooty blotch', 'soybean target spot',
         'soybean virus'],
        ['sweet potato bacterial wilt', 'sweet potato cercospora leaf spot', 'sweet potato leaf spot',
         'sweet potato scab', 'sweet potato sooty blotch', 'sweet potato virus'],
        ['wheat bacterial leaf blight', 'wheat brown rust', 'wheat crustose leaf blight', 'wheat healthy',
         'wheat leaf blight', 'wheat leaf rust', 'wheat loose smut', 'wheat mosaic disease', 'wheat powdery mildew',
         'wheat scab', 'wheat septoria blotch', 'wheat sharp eyespot', 'wheat spindle streak mosaic disease',
         'wheat spot blight', 'wheat stem rust', 'wheat stripe leaf blight', 'wheat stripe mosaic disease',
         'wheat stripe rust', 'wheat take all', 'wheat yellow leaf spot blight', 'wheat yellow rust',
         'wheat yellow stripe rust']]

    # ?????????json???
    json_file_path = 'pesticides_info_final.json'  # ?�I????JSON???��??

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels_grade = ['general', 'health', 'seriously']

    labels_prompt = ['fusiform lesions', 'irregular lesions', 'perforate lesions', 'spot lesions']

    # ???????????
    model_plant = ResNet18().to(device)
    model_plant.load_state_dict(torch.load(f'./weights/ResNet18_main_plant.pth'))
    model_plant.eval()

    model_Corn = ResNet18().to(device)
    model_Corn.load_state_dict(torch.load(f'./weights/ResNet18_disease_Corn.pth'))
    model_Corn.eval()
    model_Potato = ResNet18().to(device)
    model_Potato.load_state_dict(torch.load(f'./weights/ResNet18_disease_Potato.pth'))
    model_Potato.eval()
    model_Rice = ResNet18().to(device)
    model_Rice.load_state_dict(torch.load(f'./weights/ResNet18_disease_Rice.pth'))
    model_Rice.eval()
    model_Sorghum = ResNet18().to(device)
    model_Sorghum.load_state_dict(torch.load(f'./weights/ResNet18_disease_Sorghum.pth'))
    model_Sorghum.eval()
    model_Soybean = ResNet18().to(device)
    model_Soybean.load_state_dict(torch.load(f'./weights/ResNet18_disease_Soybean.pth'))
    model_Soybean.eval()
    model_Sweet_Potato = ResNet18().to(device)
    model_Sweet_Potato.load_state_dict(torch.load(f'./weights/ResNet18_disease_Sweet_Potato.pth'))
    model_Sweet_Potato.eval()
    model_Wheat = ResNet18().to(device)
    model_Wheat.load_state_dict(torch.load(f'./weights/ResNet18_disease_Wheat.pth'))
    model_Wheat.eval()

    model_grade = ResNet18().to(device)
    model_grade.load_state_dict(torch.load(f'./weights/ResNet18_grade.pth'))
    model_grade.eval()

    model_prompt = ResNet18().to(device)
    model_prompt.load_state_dict(torch.load(f'./weights/ResNet18_prompt.pth'))
    model_prompt.eval()
    
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    # Print the image shape to check if it's valid
    print(f"Preprocessed image shape: {image.shape}")
    
    with torch.no_grad():
        plant = model_plant(image)
    _, predicted_plant = torch.max(plant, 1)
    predicted_plant_idx = predicted_plant.item()
    predicted_plant_label = labels_plant[predicted_plant_idx]

    predicted_disease_idx = 0
    

    # ????????????????????????????
    if predicted_plant_idx == 0:
        with torch.no_grad():
            disease = model_Corn(image)
        _, predicted_disease = torch.max(disease, 1)
        predicted_disease_idx = predicted_disease.item()
    elif predicted_plant_idx == 2:
        with torch.no_grad():
            disease = model_Potato(image)
        _, predicted_disease = torch.max(disease, 1)
        predicted_disease_idx = predicted_disease.item()
    elif predicted_plant_idx == 3:
        with torch.no_grad():
            disease = model_Rice(image)
        _, predicted_disease = torch.max(disease, 1)
        predicted_disease_idx = predicted_disease.item()
    elif predicted_plant_idx == 4:
        with torch.no_grad():
            disease = model_Sorghum(image)
        _, predicted_disease = torch.max(disease, 1)
        predicted_disease_idx = predicted_disease.item()
    elif predicted_plant_idx == 5:
        with torch.no_grad():
            disease = model_Soybean(image)
        _, predicted_disease = torch.max(disease, 1)
        predicted_disease_idx = predicted_disease.item()
    elif predicted_plant_idx == 6:
        with torch.no_grad():
            disease = model_Sweet_Potato(image)
        _, predicted_disease = torch.max(disease, 1)
        predicted_disease_idx = predicted_disease.item()
    elif predicted_plant_idx == 7:
        with torch.no_grad():
            disease = model_Wheat(image)
        _, predicted_disease = torch.max(disease, 1)
        predicted_disease_idx = predicted_disease.item()
    
    predicted_disease_label = 'potato late blight'  # ????????
    try:
        predicted_disease_label = labels_disease[predicted_plant_idx][predicted_disease_idx]
        print(f"Predicted disease label: {predicted_disease_label}")
    except IndexError:
        print("Warning: Invalid indices for labels. Skipping this prediction.")
    
    text_prompt1 = 'leaf'
    text_prompt2 = 'leaf'
    level = "1"
    predicted_level = "1"  # Initialize predicted_level with a default value
    leaf_masks = []  # Initialize leaf_masks as an empty list
    with torch.no_grad():
        grade = model_grade(image)
    _, predicted_grade = torch.max(grade, 1)
    predicted_grade_idx = predicted_grade.item()
    predicted_grade_label = labels_grade[predicted_grade_idx]
    print(f"Predicted grade label: {predicted_grade_label}")

    with torch.no_grad():
        prompt = model_prompt(image)
    _, predicted_prompt = torch.max(prompt, 1)
    predicted_prompt_idx = predicted_prompt.item()
    predicted_prompt_label = labels_prompt[predicted_prompt_idx]
    print(f"Predicted prompt label: {predicted_prompt_label}")

    text_prompt2 = predicted_prompt_label
    
    #model.load_state_dict(torch.load(state_dict_path))
    image_source, image = load_image(image_path)
    image_copy = image.clone()  # Save a copy of the image tensor
    image_name = os.path.basename(image_path)
    

    sam_version = "vit_b"
    sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
    device = "cuda"
    text_prompts = [text_prompt1, text_prompt2]
    all_boxes_filt = []
    all_pred_phrases = []
    if predicted_grade_label == 'seriously':
        level = "9"
    elif predicted_grade_label == 'health':
        level = "0"
    else:
        for idx, text_prompt in enumerate(text_prompts):
            # ????????????????????????
            if 'potato' or 'tomato' in image_name and text_prompt == text_prompt2:
                model_weights = model_weights3
                # box_threshold=0.9
                # text_threshold=0.2
            else:
                model_weights = model_weights1 if text_prompt == "leaf" else model_weights2
    
            model = load_model(model_config, model_weights)
            # Create a new output directory for each prompt
            prompt_output_dir = os.path.join(image_output_dir, f"prompt_{idx}")
            os.makedirs(prompt_output_dir, exist_ok=True)
    
            boxes, logits, phrases = predict(
                model=model,
                image=image_copy,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            all_boxes_filt.extend(boxes)
            all_pred_phrases.extend(phrases)
            # Check if there are no detections after NMS
            if len(boxes) == 0:
                continue  # Skip further processing if there are no detections
    
            # print(f"Original boxes size {boxes.shape}")
            boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
            # print(f"NMS boxes size {boxes.shape}")
            print(len(boxes))
    
            # Check if there are no detections after NMS
            if len(boxes) == 0:
                continue  # Skip further processing if there are no detections
    
            predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    
            # ????
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            image_name = os.path.basename(image_path)
    
    
            # ?��?????
            H, W = image_source.shape[0], image_source.shape[1]
            for i in range(boxes.size(0)):  # boxes_filt.size(0)
                boxes[i] = boxes[i] * torch.Tensor([W, H, W, H]).to(boxes[i].device)
                boxes[i][:2] -= boxes[i][2:] / 2
                boxes[i][2:] += boxes[i][:2]
            boxes = boxes.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(device)
    
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )
            # If this is the "leaf" prompt, save the masks of "leaf" instances
            # ??????????????
            # predicted_level = "1"
            if text_prompt == text_prompt1:
                leaf_masks = masks
    
            # If this is the "spot" prompt, calculate the proportion of "spot" pixels in each "leaf" instance
            elif text_prompt == text_prompt2:
                for leaf_mask in leaf_masks:
                    leaf_area = torch.sum(leaf_mask).item()
                    total_common_pixels = torch.zeros_like(leaf_mask)
                    for mask in masks:
                        common_pixels = leaf_mask & mask
                        total_common_pixels += common_pixels
                    proportion = torch.sum(total_common_pixels).item() / leaf_area
                    print(f"The total proportion of {text_prompt} pixels in {image_name} area is {proportion}")
                    if proportion <= 0.10:
                        level = "1"
                        print(f"The degree of disease in {image_name} is Level {level}")
                    elif 0.11 <= proportion <= 0.25:
                        level = "3"
                        print(f"The degree of disease in {image_name} is Level {level}")
                    elif 0.26 <= proportion <= 0.40:
                        level = "5"
                        print(f"The degree of disease in {image_name} is Level {level}")
                    elif 0.41 <= proportion <= 0.65:
                        level = "7"
                        print(f"The degree of disease in {image_name} is Level {level}")
                    elif proportion >= 0.66:
                        level = "9"
                        print(f"The degree of disease in {image_name} is Level {level}")
            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes, phrases):
                show_box(box.numpy(), plt.gca(), label)
    
            plt.axis('off')
            plt.savefig(
                os.path.join(prompt_output_dir, "grounded_sam_output.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
        
    predicted_level = level  #??????????

    return predicted_level


def test_accuracy(model_weights1, model_weights2, model_weights3, image_dir, output_dir, text_prompt1, text_prompt2, box_threshold, text_threshold):
    # ?????????��??
    image_paths = glob.glob(os.path.join(image_dir, '*/*.*'))

    correct_count = 0
    total_count = 0
    # ??????��?
    incorrect_images_info = []

    # ????????????????????
    level_stats = {str(level): {'correct': 0, 'total': 0} for level in [0, 1, 3, 5, 7, 9]}
    
    # ????????????��???????5??7????
    level_1_output_dir = os.path.join(output_dir, 'level_1')
    level_3_output_dir = os.path.join(output_dir, 'level_3')
    level_5_output_dir = os.path.join(output_dir, 'level_5')
    level_7_output_dir = os.path.join(output_dir, 'level_7')
    level_9_output_dir = os.path.join(output_dir, 'level_9')
    level_0_output_dir = os.path.join(output_dir, 'level_0')
    os.makedirs(level_1_output_dir, exist_ok=True)
    os.makedirs(level_3_output_dir, exist_ok=True)
    os.makedirs(level_5_output_dir, exist_ok=True)
    os.makedirs(level_7_output_dir, exist_ok=True)
    os.makedirs(level_9_output_dir, exist_ok=True)
    os.makedirs(level_0_output_dir, exist_ok=True)

    for image_path in image_paths:
        # ??????????????????????????
        true_label = os.path.basename(os.path.dirname(image_path))
        image_source, image = load_image(image_path)
        image_source = Image.fromarray(image_source)
        
        # ???? process_image ????
        predicted_level = process_image(
            image_path=image_path,
            model_weights1=model_weights1,
            model_weights2=model_weights2,
            model_weights3=model_weights3,
            text_prompt1=text_prompt1,
            text_prompt2=text_prompt2,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            image_output_dir=os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0])
        )

        # ????????????????
        if predicted_level == true_label:
            correct_count += 1
        else:
            # ??????????????????????????????
            incorrect_images_info.append((os.path.basename(image_path), predicted_level, true_label))
        
        total_count += 1
        
        # ??????????
        level_stats[true_label]['total'] += 1
        if predicted_level == true_label:
            level_stats[true_label]['correct'] += 1

        # ????????????????
        if predicted_level == '1':
            shutil.copy(image_path, os.path.join(level_1_output_dir, os.path.basename(image_path)))
        elif predicted_level == '3':
            shutil.copy(image_path, os.path.join(level_3_output_dir, os.path.basename(image_path)))
        elif predicted_level == '5':
            shutil.copy(image_path, os.path.join(level_5_output_dir, os.path.basename(image_path)))
        elif predicted_level == '7':
            shutil.copy(image_path, os.path.join(level_7_output_dir, os.path.basename(image_path)))
        elif predicted_level == '9':
            shutil.copy(image_path, os.path.join(level_9_output_dir, os.path.basename(image_path)))
        elif predicted_level == '0':
            shutil.copy(image_path, os.path.join(level_0_output_dir, os.path.basename(image_path)))

    # ??????
    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy:.4f}")
    
    # ?????????????
    for level, stats in level_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"Level {level}: Accuracy = {accuracy:.4f} (Correct: {stats['correct']}, Total: {stats['total']})")
        else:
            print(f"Level {level}: No data available.")
        
if __name__ == "__main__":
    model_weights1 = "./weights/groundingdino_swint_ogc.pth"
    model_weights2 = "./weights/model_weights810.pth"
    model_weights3 = "./weights/model_weights200.pth"
    output_dir = "./allstable_out"
    image_dir = "./allstable"
    box_threshold = 0.45
    text_threshold = 0.4
    text_prompt1 = "leaf"
    text_prompt2 = "leaf"

    test_accuracy(
        model_weights1=model_weights1,
        model_weights2=model_weights2,
        model_weights3=model_weights3,
        image_dir=image_dir,
        output_dir=output_dir,
        text_prompt1=text_prompt1,
        text_prompt2=text_prompt2,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
