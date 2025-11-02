# 1. jsonデータたちからtrainするためのデータを抽出,変換する
# 2. yamlファイルの書き換え

import json
import os
import re
import shutil
import yaml
import subprocess
import copy

# 変更するところ
data_ver = "2024_07_11_09_34_02"
selected_labels = ["so","io","sr","ir","lr","mr"] #["so","io","sr","ir","lr","mr"] または load_config(yaml_file) を使用

def extract_and_convert_labels(json_file, selected_labels):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    converted_shapes = []
    for shape in data['shapes']:
        original_label = shape['label']
        if original_label in selected_labels:
            converted_shapes.append(shape)
        elif original_label.startswith('l_') and original_label[2:] in selected_labels:
            shape['label'] = original_label[2:]  # Remove 'l_' prefix
            converted_shapes.append(shape)
        elif original_label.startswith('r_') and original_label[2:] in selected_labels:
            shape['label'] = original_label[2:]  # Remove 'r_' prefix
            converted_shapes.append(shape)
    
    data['shapes'] = converted_shapes
    return data

def create_empty_json(json_file):
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    return {
        "version": "4.5.9",
        "flags": {},
        "shapes": [],
        "imagePath": f"{base_name}.tiff",
        "imageData": None,
        "imageHeight": 480,
        "imageWidth": 480
    }

def process_json_files(json_dir, output_dir, selected_labels):
    os.makedirs(output_dir, exist_ok=True)
    
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            filtered_data = extract_and_convert_labels(json_path, selected_labels)
            
            if not filtered_data['shapes']:
                filtered_data = create_empty_json(json_file)
            
            output_file = os.path.join(output_dir, json_file)
            with open(output_file, 'w') as f:
                json.dump(filtered_data, f, indent=2)

def copy_corresponding_images(json_dir, output_dir, image_extensions=['.tiff', '.jpg', '.jpeg', '.png']):
    for json_file in os.listdir(output_dir):
        if json_file.endswith('.json'):
            base_name = os.path.splitext(json_file)[0]
            for ext in image_extensions:
                image_file = base_name + ext
                src_path = os.path.join(json_dir, image_file)
                if os.path.exists(src_path):
                    dst_path = os.path.join(output_dir, image_file)
                    shutil.copy2(src_path, dst_path)
                    break
            else:
                print(f"No corresponding image found for {json_file}")

def find_next_available_number(base_path, current_num):
    while True:
        new_path = re.sub(r'\d+$', str(current_num), base_path)
        if not os.path.exists(new_path):
            return new_path
        current_num += 1

def update_directory_name(path):
    # ディレクトリ名を取得
    dir_name = os.path.basename(path)
    
    # 正規表現を使用して末尾の数字を抽出
    match = re.search(r'(\d+)$', dir_name)
    
    if match:
        last_num = int(match.group(1))
        # 4以上の場合、次に利用可能な番号を見つける
        return find_next_available_number(path, last_num + 1)
    else:
        # 末尾に数字がない場合は変更しない
        return path

def run_labelme2yolo(json_dir):
    command = f"labelme2yolo --json_dir {json_dir}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("labelme2yolo command executed successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing labelme2yolo command:")
        print(e.stderr)

class PreserveFormatDumper(yaml.Dumper):
    def represent_list(self, data):
        # リストがすべて文字列の場合、フロースタイルで表現
        if all(isinstance(item, str) for item in data):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return super().represent_list(data)

def copy_yaml_content(source_file, target_file, path):
    # ソースファイルから必要な内容を読み取る
    with open(source_file, 'r') as file:
        source_data = yaml.safe_load(file)
    
    # 必要な情報を抽出
    nc = source_data.get('nc')
    names = source_data.get('names')
    
    if nc is None or names is None:
        print(f"Required content not found in {source_file}")
        return
    
    # ターゲットファイルに内容をコピー
    try:
        with open(target_file, 'r') as file:
            target_data = yaml.safe_load(file)
        
        # ディープコピーを作成
        updated_data = copy.deepcopy(target_data)
        
        # ncとnames、pathを更新
        updated_data['nc'] = nc
        updated_data['names'] = names  # これは既に配列形式です
        updated_data['path'] = path
        
        # 更新した内容を書き込む
        with open(target_file, 'w') as file:
            yaml.dump(updated_data, file, Dumper=PreserveFormatDumper, default_flow_style=False, sort_keys=False)
        
        print(f"Successfully updated nc, names, and path in {target_file}")
    except Exception as e:
        print(f"Error updating {target_file}: {str(e)}")

# Usage
output_dir = 'datasets/'+data_ver+'/filtered_json1'
output_dir = update_directory_name(output_dir) #datasets/2024_07_11_09_34_02/filtered_json1234
print(output_dir)
yaml_file = 'dataset.yaml'
path_from_yaml = output_dir[9:] + "/YOLODataset/" #2024_07_11_09_34_02/filtered_json1234 + /YOLODataset/
output_yaml=output_dir+"/YOLODataset/dataset.yaml"
json_dir = 'datasets/'+data_ver+'/Train'

process_json_files(json_dir, output_dir, selected_labels)
copy_corresponding_images(json_dir, output_dir)
run_labelme2yolo(output_dir)
copy_yaml_content(output_yaml, yaml_file, path_from_yaml)