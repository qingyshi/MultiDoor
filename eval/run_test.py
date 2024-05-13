import os
import cv2
import torch


test_data_dir = "/data00/multidoor_dataset/test_case"
for test_cases in os.listdir(test_data_dir):
    test_cases_path = os.path.join(test_data_dir, test_cases)
    num_case = len(os.listdir(test_cases_path))
    if num_case <= 1:
        continue
    
    for single_case in os.listdir(test_cases_path):
        single_case_path = os.path.join(test_cases_path, single_case)
        bg_image_path = os.path.join(single_case_path, "bg.jpg")
        bg_mask_path = [os.path.join(single_case_path, tar) for tar in ["tar1.jpg", "tar2.jpg"]]
        
        save_id = 0
        for another_case in os.listdir(test_cases_path):
            save_path = os.path.join(single_case_path, f"gen_{another_case}.jpg")
            if another_case == single_case:
                continue
            else:
                ref_image_path = [os.path.join(test_cases_path, another_case, ref) for ref in ["ref1.jpg", "ref2.jpg"]]
            

            