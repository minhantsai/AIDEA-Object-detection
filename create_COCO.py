import pandas as pd
import numpy as np
import json
import datetime

df = pd.read_json('Matt/data.json')

### info
info = {
    "description":"This is 1.0 version of the AIDEA competition dataset in COCO format create by Team MAKOBY.",
    "url":"https://aidea-web.tw/",
    "version":"1.0",
    "year":2020,
    "contributor":"MHTsai, member of MAKOBY.",
    "date_created": datetime.datetime(2021, 2, 21),
}

### license
licenses = [{
    "id":1,    
    "name":"Free License",
    "url":"https://aidea-web.tw/topic/35e0ddb9-d54b-40b7-b445-67d627890454",
}]

### proto for image, annotation, category
image_proto = {
    "file_name":"COCO_val2014_000000391895.jpg",
    "id":391895,
    "license":1,
    "height":1080,
    "width":1920,
    "flickr_url":"",
    "coco_url":"",
    "date_captured": datetime.datetime(2021, 2, 21),
}

annotation_proto = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "area": float,
    "bbox": "[x,y,width,height]",
    "iscrowd": 0,
    "segmentation": [],
}

category_proto = {
    "id": int,
    "name": str,
    "supercategory": "AIDEA",
}

### category
label = ['vehicle','pedestrian','scooter','bicycle']
label_dict = {
    'vehicle': 1,
    'pedestrian': 2,
    'scooter': 3,
    'bicycle': 4,
}

categories = list()
for idx in range(1,5):
    category = category_proto.copy()
    category['id'] = idx
    category['name'] = label[idx-1]
    categories.append(category)
    
### annotations and images
images, annotations = list(), list()
image_id = 1
annotation_id = 1

for index, row in df.iterrows():
    # for image 
    image = image_proto.copy()
    image['file_name'] = row['filename']
    image['id'] = image_id
    
    
    # for annotation
    VOC_bboxes = np.array(df.iloc[index]['boxes']).astype(np.float)
    bboxes = list(map(lambda s: [s[0],s[1],s[2]-s[0],s[3]-s[1]] , VOC_bboxes))
    labels = list(map(lambda s: label_dict[s], df.iloc[index]['labels']))
    for bbox, label in zip(bboxes,labels):
        annotation = annotation_proto.copy()
        annotation['bbox'] = bbox
        annotation['category_id'] = label
        annotation['area'] = bbox[2]*bbox[3]
        annotation['id'] = annotation_id
        annotation['image_id'] = image_id
        annotation_id = annotation_id + 1
        annotations.append(annotation)
    
    image_id = image_id + 1
    images.append(image)
    
### final
AIDEA_COCO = {
    "info": info,
    "liceses": licenses,
    "categories": categories,
    "images": images,
    "annotations": annotations
}

## create COCO json file
def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

with open('./Matt/COCO_data.json', 'w') as fp:
    json.dump(AIDEA_COCO, fp, default = myconverter)