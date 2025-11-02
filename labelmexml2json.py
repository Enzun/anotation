import os
import json
import xml.etree.ElementTree as ET
import glob
import shutil

# name of downloaded dir 簡単な名前に変えてからのほうがいいよ
dir_name ='2024_07_11_09_34_02/'
dirdirname = 'Train'

xml_dir = "DATA/xmls/"+dir_name + dirdirname
json_dir = "datasets/"+dir_name #出力先
img_dir = "DATA/MRI_IMGs/"

def xml_to_json(xml_path, json_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = {
        "version": "4.5.9",
        "flags": {},
        "shapes": [],
        "imagePath": root.find("filename").text,
        "imageData": None,
        "imageHeight": int(root.find("imagesize/nrows").text),
        "imageWidth": int(root.find("imagesize/ncols").text)
    }
    
    for obj in root.findall("object"):
        label = obj.find("name").text
        
        # ポリゴンの処理
        polygon = obj.find("polygon")
        if polygon is not None:
            points = []
            for pt in polygon.findall("pt"):
                x = float(pt.find("x").text)
                y = float(pt.find("y").text)
                points.append([x, y])
            
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            data["shapes"].append(shape)
        
        # バウンディングボックスの処理
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            shape = {
                "label": label,
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            data["shapes"].append(shape)
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def copyIMG(img_dir,img_name, json_dir):
    ex = glob.glob(os.path.join(img_dir, "**", img_name), recursive=True)
    for i in range(len(ex)):
        file=ex[i]
        # print(file)
        shutil.copy(file,json_dir)


os.makedirs(json_dir, exist_ok=True)

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, xml_file)
        json_path = os.path.join(json_dir, xml_file.replace(".xml", ".json"))
        img_name = xml_file.replace(".xml", ".tiff")

        xml_to_json(xml_path, json_path)
        # print("Conversion from XML to JSON completed.")
        copyIMG(img_dir,img_name,json_dir)        
        # print("Copied images from "+img_dir+" to " +json_dir+".")

