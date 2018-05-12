import cv2
import numpy as np
import cPickle as pickle
import os

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

    '''
    Extracts the feature vector for one image
    '''
    def get_feature_vector(self, image_path, vector_size=32):
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

    '''
    Calls the get_feature_vector for all images specified in images_path and
    writes to the file specified in features_file_path
    '''
    def extract(self):
        files = [os.path.join(self.images_path, p) for p in sorted(os.listdir(self.images_path))]

        with open(self.features_file_path, 'w') as fp:
            for f in files:
                print 'Extracting features from image %s' % f
                name = f.split('/')[-1].lower()
                result = self.get_feature_vector(f)
                fp.write(name+','+','.join(str(e) for e in result)+'\n')

def run():
    featureExtraction = Feature()
    featureExtraction.extract()
run()
