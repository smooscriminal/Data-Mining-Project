'''
The purpose of this script is to read the labels from the json file provided by yelp and add
the label to our features file.
'''
import json
import os

'''
Gets the data from the yelp json file and builds a dictionary with the file_name as the
key and the label as the value.
'''
def get_labels(file_name="./yelp_photos/yelp_photos/photos.json"):
    image_labels = {}
    with open(file_name, "r") as fp_r:
        for line in fp_r:
            image_obj=json.loads(line)
            key = image_obj["photo_id"].upper()
            value = image_obj["label"]
            image_labels[key] = value
    return image_labels

'''
creates a new feature file with the labels. It takes a dictionary object as one
of the parameters so it can search for the appropriate labels. Also takes a file_name
which is the file the function will be reading from.
'''
def add_labels_to_features(lookup, limit_to, file_name="features.pck"):
    with open(file_name, "r") as fp_r, open("features_with_labels.csv", "w") as fp_w:
        for line in fp_r:
            values=line.split(",")
            key=(os.path.splitext(values[0])[0]).upper()
            label = lookup.get(key)
            if limit_to.get(label) != None:
                fp_w.write(line.strip()+","+label+"\n")

extract = {"food": 1, "drink":2}
lookup = get_labels()
add_labels_to_features(lookup, extract)
