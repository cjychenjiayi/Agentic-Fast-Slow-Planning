import os
import json
import cv2

def resize_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = x1 / 1000 * width
    y1 = y1 / 1000 * height
    x2 = x2 / 1000 * width
    y2 = y2 / 1000 * height
    bbox = [x1, y1, x2, y2]
    return [int(b) for b in bbox]

def parse_data_jsonl(input_string:str, image_path = None):
    width = height = None
    if image_path is not None:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
    lines = input_string.split('\n')
    data_list = []
    
    for i in range(0, len(lines), 2):
        first_line = lines[i]
        ref_part, box_part = first_line.split('<box>')
        ref = ref_part.replace('<ref>', '').replace("</ref>", "").strip()
        box = box_part.replace("</box>", "").replace("[", "").replace("]", "").split(',')
        box = [int(coord) for coord in box]
        
        second_line = lines[i + 1]
        dist_part, orient_part = second_line.split(',')
        dist = float(dist_part.split(':')[1].strip())
        orient = float(orient_part.split(':')[1].strip())
        if width is not None:
            box = resize_bbox(box, width, height)
        data_dict = {
            'ref': ref,
            'box': box,
            'dist': dist,
            'orient': orient
        }
        data_list.append(data_dict)
    return data_list

def parse_data_internvl(input_string:str, image_path = None):
    width = height = None
    if image_path is not None:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
    lines = input_string.split('\n')
    data_list = []
    
    for i in range(0, len(lines), 2):
        try:
            print(lines[i], lines[i+1])
            first_line = lines[i]
            ref_part, box_part = first_line.split('<box>')
            ref = ref_part.replace('<ref>', '').replace("</ref>", "").strip()
            box = box_part.replace("</box>", "").replace(r"[", "").replace(r"]", "").split(',')
            box = [int(coord) for coord in box]
            
            second_line = lines[i + 1]
            dist_part, orient_part = second_line.split(',')
            dist = float(dist_part.split(':')[1].strip())
            orient = float(orient_part.split(':')[1].strip())
            if width is not None:
                box = resize_bbox(box, width, height)
            data_dict = {
                'ref': ref,
                'box': box,
                'dist': dist,
                'orient': orient
            }
            data_list.append(data_dict)
        except:
            pass
    return data_list

def visualize(bbox_dict, image_path, need_resize=False, output_folder="Result"):
    image=cv2.imread(image_path);h,w,_=image.shape
    color_box=(0,200,180);color_text=(255,255,255);color_bg=(30,30,60)
    font=cv2.FONT_HERSHEY_SIMPLEX;font_scale=1.2;font_thickness=2
    for bbox,type_name in bbox_dict:
        if need_resize:bbox=resize_bbox(bbox,w,h)
        x1,y1,x2,y2=bbox
        cv2.rectangle(image,(x1,y1),(x2,y2),color_box,5)
        text=str(type_name);(tw,th),_=cv2.getTextSize(text,font,font_scale,font_thickness)
        tx,ty=x1,(y1-10 if y1-10>th else y2+th+10)
        cv2.rectangle(image,(tx,ty-th),(tx+tw,ty),color_bg,-1)
        cv2.putText(image,text,(tx,ty-2),font,font_scale,color_text,font_thickness,cv2.LINE_AA)
    os.makedirs(output_folder,exist_ok=True)
    cv2.imwrite(os.path.join(output_folder,os.path.basename(image_path)),image)



def parse_jsonl(jsonl_file_path, replace_folder = None):
    result = []
    
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                image = data.get("image")
                if image is not None and replace_folder is not None:
                    image = os.path.join(replace_folder, os.path.basename(image))
                    
                gpt_value = None
                for convo in data.get("conversations", []):
                    if convo.get("from") == "gpt":
                        gpt_value = convo.get("value")
                        gpt_value = parse_data_jsonl(gpt_value, image)
                        break
                if image and gpt_value:
                    result.append({
                        "image": image,
                        "gpt_value": gpt_value
                    })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    return result


if __name__ == "__main__":
    content = 'vehicle[[491, 509, 513, 538]]\nDist: 39.3, Orient: 0\ntable[[0, 590, 282, 766]]\nDist: 5.9, Orient: 35'
    print(parse_data_internvl(content))