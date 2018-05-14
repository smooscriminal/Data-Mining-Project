import cv2
import numpy as np
import cPickle as pickle
import os
import json
from shutil import copyfile

'''
This object extracts the features from images in a given path. It then writes
the features to a file named features.pck in the same location as this file if
no value is passed in for features_file_path. Also processes the images in path
./yelp_photos/photos/ by default if nothing is passed in for images_path.
'''
class Feature(object):
    def __init__(self, features_file_path='features.pck', images_path="./yelp_photos/yelp_photos/photos/" ):
        self.features_file_path = features_file_path
        self.images_path = images_path
        max_number_img = 4000
        self.images = np.empty(shape=(max_number_img,2), dtype=object)
        with open("./yelp_photos/yelp_photos/photos.json", "r") as fp_r:
            index = 0
            count_food = 0
            count_drink = 0
            for line in fp_r:
                image_obj=json.loads(line)
                if (image_obj["label"] == "food" and count_food > 1999 ):
                    continue
                elif (image_obj["label"] == "drink" and count_drink > 1999 ):
                    continue
                elif image_obj["label"] == "food":
                    count_food = count_food + 1
                elif image_obj["label"] == "drink":
                    count_drink = count_drink + 1
                else:
                    continue
                self.images[index][0] = image_obj["photo_id"]+".jpg"
                self.images[index][1] = image_obj["label"]
                if count_food > 1999 and count_drink > 1999:
                    print "done"
                    break
                index = index + 1
    '''
    Extracts the feature vector for one image
    '''
    def get_feature_vector_kaze(self, image_path, vector_size=32):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        try:
            alg = cv2.KAZE_create()
            key_points = alg.detect(image) # All key points
            # Reduces top vector_size number of key points. I read that larger is better
            key_points = sorted(key_points, key=lambda x: -x.response)[:vector_size]
            key_points, descriptor = alg.compute(image,key_points)
            descriptor = descriptor.flatten()
            expected_size = (vector_size * 64)
            if descriptor.size < expected_size:
                descriptor = np.concatenate([descriptor, np.zeros(expected_size - descriptor.size)])
        except cv2.error as e:
            print 'Error: ', e
            return None
        return descriptor

    def get_feature_vector_sift(self, image_path):
        """
        Generate SIFT features for images
        Parameters:
        -----------
        labeled_img_paths : list of lists
            Of the form [[image_path, label], ...]
        Returns:
        --------
        img_descs : list of SIFT descriptors with same indicies as labeled_img_paths
        y : list of corresponding labels
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        desc = (sift.detectAndCompute(gray, None))[1]
        return desc

    def move_sample_set(self):
        dir = './sample_data_set'
        if (not os.path.exists(dir)):
            os.makedirs(dir)
        if (not os.path.exists(dir+'/photos')):
            os.makedirs(dir+'/photos')
        dir = dir + '/'
        with open(dir+'photo_data.json','w') as fp:
            for f, label in self.images:
                photo = {}
                photo["file_name"] = f
                photo["label"] = label
                copyfile(self.images_path + f, dir + '/photos/' + f)
                print "copied image %s" % f
                fp.write(json.dumps(photo)+'\n')
        print "copy finished"
        
    '''
    Calls the get_feature_vector for all images specified in images_path and
    writes to the file specified in features_file_path
    '''
    def extract(self, algorithm, features_file_path = ""):
        if features_file_path != "":
            self.features_file_path  = features_file_path
        with open(self.features_file_path, 'w') as fp:
            for f, label in self.images:
                if str(algorithm).upper() == 'KAZE':
                    print 'Extracting features from image %s using kaze' % f
                    result = self.get_feature_vector_kaze(self.images_path + f)
                    fp.write(f+','+','.join(str(e) for e in result)+","+label+'\n')
                elif str(algorithm).upper() == 'SIFT':
                    print 'Extracting features from image %s using sift' % f
                    name = f.split('/')[-1].lower()
                    result = self.get_feature_vector_sift(self.images_path + f)
                    fp.write(f+','+','.join(str(e) for e in result)+","+label+'\n')
                    print "done"


def run():
    feature_extraction = Feature(features_file_path="KAZE.csv")
    feature_extraction.move_sample_set()
    # feature_extraction.extract(algorithm="KAZE")
    # feature_extraction.extract(algorithm="SIFT", features_file_path="SIFT.csv")
run()
