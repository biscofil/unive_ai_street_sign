import csv
import os

def getImages(path="./GTSRB-Training_fixed/GTSRB/Training"):
    out = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".csv"):
                full_path = root + "/" + name
                with open(full_path) as csv_file:
                    csv_reader = csv.DictReader(csv_file, delimiter=';', quoting=csv.QUOTE_NONE)
                    images = list(csv_reader)
                    class_id = images[0]['ClassId']
                    out[class_id] = [root + "/" + image['Filename'] for image in images]
                    # for image in images:
                    #    print(root + "/" + image['Filename'], "has class", image['ClassId'])
    return out


print(getImages())
