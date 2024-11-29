# -*- coding: utf-8 -*-
import faulthandler
faulthandler.enable()
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
import cv2
import json
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
from torchvision import transforms
import numpy as np
from ResNet_18_34 import ResNet18
import shutil
import torch.nn.functional as F 


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

def process_single_image(
        model_config="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights1="weights/groundingdino_swint_ogc.pth",
        model_weights2="weights/model_weights840.pth",
        model_weights3="weights/model_weights200.pth",
        image_path="D:/postgraduate/Grounding-Dino-FineTuning-main/.asset/test1.jpg",
        text_prompt1="leaf",
        text_prompt2="spot lesions",
        box_threshold=0.45,
        text_threshold=0.35,
        output_dir = "./outputs"
):
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

    # json文件
    json_file_path = 'pesticides_info_final.json' 

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels_grade = ['general', 'health', 'seriously']

    labels_prompt = ['fusiform lesions', 'irregular lesions', 'perforate lesions', 'spot lesions']

    # 加载模型权重
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
    # print(f"Preprocessed image shape: {image.shape}")
    
    with torch.no_grad():
        plant = model_plant(image)
    _, predicted_plant = torch.max(plant, 1)
    predicted_plant_idx = predicted_plant.item()
    predicted_plant_label = labels_plant[predicted_plant_idx]

    predicted_disease_idx = 0
    

    # 根据预测的植物种类，选择相应的疾病模型进行预测
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
    
    predicted_disease_label = 'potato late blight'  # 默认值
    try:
        predicted_disease_label = labels_disease[predicted_plant_idx][predicted_disease_idx]
        print(f"Predicted disease label: {predicted_disease_label}")
    except IndexError:
        print("Warning: Invalid indices for labels. Skipping this prediction.")
    
    text_prompt1 = 'leaf'
    text_prompt2 = 'leaf'
    level = "1"
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
        for text_prompt in text_prompts:
            if ('potato' in image_name or 'tomato' in image_name) and text_prompt == text_prompt2:
                model_weights = model_weights3
                box_threshold=0.9
                text_threshold=0.2
            elif text_prompt == text_prompt1:
                model_weights = model_weights1
            else:
                model_weights = model_weights2

            model = load_model(model_config, model_weights)

            boxes, logits, phrases = predict(
                model=model,
                image=image_copy,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            all_boxes_filt.extend(boxes)
            all_pred_phrases.extend(phrases)

            # print(f"Original boxes size {boxes.shape}")
            boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
            # print(f"NMS boxes size {boxes.shape}")

            predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

            # 加载图像
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            image_name = os.path.basename(image_path)


            # 获取图像的高度和宽度
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
            if text_prompt == text_prompt1:
                leaf_masks = masks
            # If this is the "spot" prompt, calculate the proportion of "spot" pixels in each "leaf" instance
            else:
                print("Processing 'perforate lesions' prompt")
                print(f"Lesions masks shape: {masks.shape if masks is not None else 'None'}")
                for leaf_mask in leaf_masks:
                    leaf_area = torch.sum(leaf_mask).item()
                    total_common_pixels = torch.zeros_like(leaf_mask)
                    seen_pixels = torch.zeros_like(leaf_mask, dtype=torch.bool)
                    for mask in masks:
                        common_pixels = leaf_mask & mask
                        new_common_pixels = common_pixels & ~seen_pixels
                        total_common_pixels += new_common_pixels
                        seen_pixels |= new_common_pixels

                    proportion = torch.sum(total_common_pixels).item() / leaf_area
                    print(f"The total proportion of {text_prompt} pixels in {image_name} area is {proportion}")
                    if proportion <= 0.10:
                        level = "1"
                    elif 0.11 <= proportion <= 0.25:
                        level = "3"
                    elif 0.26 <= proportion <= 0.40:
                        level = "5"
                    elif 0.41 <= proportion <= 0.65:
                        level = "7"
                    elif proportion >= 0.66:
                        level = "9"

            # 绘制并保存分割结果
            plt.figure(figsize=(10, 10))
            plt.imshow(image_source)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes, phrases):
                show_box(box.numpy(), plt.gca(), label)
            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, f"{image_name}_{text_prompt}_segmentation.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )

        # 保存叶片分割图
        if leaf_masks is not None and len(leaf_masks) > 0:
            if isinstance(leaf_masks, torch.Tensor):
                leaf_masks = [leaf_masks]  # 将单个张量转换为列表
            leaf_mask_combined = torch.sum(torch.stack(leaf_masks), dim=0).cpu().numpy()
            # 去掉前两个维度，使其成为二维数组
            leaf_mask_combined = leaf_mask_combined.squeeze()
            leaf_mask_combined = (leaf_mask_combined > 0).astype(np.uint8) * 255
            leaf_mask_image = Image.fromarray(leaf_mask_combined)
            leaf_mask_image.save(os.path.join(output_dir, f"{image_name}_leaf_mask.png"))

        # 保存病灶分割图
        if masks is not None and len(masks) > 0:
            if isinstance(masks, torch.Tensor):
                masks = [masks]  # 将单个张量转换为列表
            spot_mask_combined = torch.sum(torch.stack(masks), dim=0).cpu().numpy()
            # 去掉第一个维度，使其成为三维数组
            spot_mask_combined = spot_mask_combined.sum(axis=0)
            # 去掉第一个维度，使其成为二维数组
            spot_mask_combined = spot_mask_combined.squeeze()
            spot_mask_combined = (spot_mask_combined > 0).astype(np.uint8) * 255
            spot_mask_image = Image.fromarray(spot_mask_combined)
            spot_mask_image.save(os.path.join(output_dir, f"{image_name}_spot_mask.png"))

    #保存预测结果到文本文件
    result_text = (
        f"Image: {image_name}\n"
        f"Plant Type: {predicted_plant_label}\n"
        f"Disease Type: {predicted_disease_label}\n"
        f"Initial Severity: {predicted_grade_label}\n"
        f"Disease Prompt: {predicted_prompt_label}\n"
        f"Final Severity Level: {level}\n"
    )
    with open(os.path.join(output_dir, f"{image_name}_result.txt"), 'w') as f:
        f.write(result_text)

    return level

if __name__ == "__main__":
    model_weights1 = "./weights/groundingdino_swint_ogc.pth"
    model_weights2 = "./weights/model_weights810.pth"
    model_weights3 = "./weights/model_weights200.pth"
    output_dir = "./outputs"
    image_path = "/home/wuxingcai/ChatGPT/main/potato_new/3/potato_early_blight_118.jpg"
    box_threshold = 0.45
    text_threshold = 0.4
    text_prompt1 = "leaf"
    text_prompt2 = "spot"

    os.makedirs(output_dir, exist_ok=True)

    final_severity_level= process_single_image(
        image_path=image_path,
        model_weights1=model_weights1,
        model_weights2=model_weights2,
        model_weights3=model_weights3,
        text_prompt1=text_prompt1,
        text_prompt2=text_prompt2,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output_dir=output_dir
    )

    print(f"Final severity level for {os.path.basename(image_path)}: {final_severity_level}")