import json
import requests

import time

def call_vision_api(image_filename, api_keys):
    api_key = api_keys['microsoft']

    post_url = "https://westeurope.api.cognitive.microsoft.com/vision/v1.0/analyze?visualFeatures=Categories,Tags,Description,Faces,ImageType,Color,Adult"
    image_data = open(image_filename, 'rb').read()
    result = requests.post(post_url, data=image_data, headers={'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': api_key,})
    result.raise_for_status()
    response = json.loads(result.text)

    post_url = "https://westeurope.api.cognitive.microsoft.com/vision/v1.0/ocr?language=unk"
    image_data = open(image_filename, 'rb').read()
    result = requests.post(post_url, data=image_data, headers={'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': api_key, })
    result.raise_for_status()
    response_ = json.loads(result.text)

    print(response_)

    response = dict(response.items() + response_.items())

    time.sleep(1)

    return response

# Return a dictionary of features to their scored values (represented as lists of tuples).
# Scored values must be sorted in descending order.
#
# {
#    'feature_1' : [(element, score), ...],
#    'feature_2' : ...
# }
#
# E.g.,
#
# {
#    'tags' : [('throne', 0.95), ('swords', 0.84)],
#    'description' : [('A throne made of pointy objects', 0.73)]
# }
#
def get_standardized_result(api_result):
    output = {
        'tags' : [],
        'captions' : [],
#        'categories' : [],
#        'adult' : [],
#        'image_types' : []
#        'tags_without_score' : {}
    }

    for tag_data in api_result['tags']:
        output['tags'].append((tag_data['name'], tag_data['confidence']))

    for caption in api_result['description']['captions']:
        output['captions'].append((caption['text'], caption['confidence']))

#    for category in api_result['categories']:
#        output['categories'].append(([category['name'], category['score']))

#    output['adult'] = api_result['adult']

#    for tag in api_result['description']['tags']:
#        output['tags_without_score'][tag] = 'n/a'

#    output['image_types'] = api_result['imageType']

    if 'regions' in api_result:
        output['text_tags'] = []
        for region in api_result['regions']:
            #print(region)
            for line in region['lines']:
                #print(line)
                for word in line['words']:
                    #print(word)
                    output['text_tags'].append((word['text'], 1.0))

    return output
