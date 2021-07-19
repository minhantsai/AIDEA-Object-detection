import json
import pandas as pd

data = pd.read_json('./data.json')

# {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}

result_proto = {
    "image_id": 123,
    "category_id": 1,
    "bbox": [],
    "score": 0
}

label = ['vehicle','pedestrian','scooter','bicycle']
label_dict = {
    'vehicle': 1,
    'pedestrian': 2,
    'scooter': 3,
    'bicycle': 4,
}

results = list()
image_id = 1

for index, row in data.iterrows():
    
    file_name = row['filename']
    id = image_id
    
    # for result
    VOC_bboxes = np.array(data.iloc[index]['boxes']).astype(np.float)
    bboxes = list(map(lambda s: [s[0],s[1],s[2]-s[0],s[3]-s[1]] , VOC_bboxes))
    labels = list(map(lambda s: label_dict[s], data.iloc[index]['labels']))
    for bbox, label in zip(bboxes,labels):
        result = result_proto.copy()
        result['image_id'] = id
        result['category_id'] = label
        result['bbox'] = bbox
        result['score'] = 0.5
        results.append(result)
    
    image_id = image_id + 1
    
with open('resfile.json', 'w') as jsonfile:
    json.dump(results, jsonfile)