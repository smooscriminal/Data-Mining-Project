import cv2
import numpy as np
import cPickle as pickle
import os

class Feature(object):
    def __init__(self, features_file_path='features.pck', images_path='./yelp_photos/photos/' ):
        self.features_file_path = features_file_path
        self.images_path = images_path

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

    def extract(self):
        files = [os.path.join(self.images_path, p) for p in sorted(os.listdir(self.images_path))]

        result = {}
        for f in files:
            print 'Extracting features from image %s' % f
            name = f.split('/')[-1].lower()
            result[name] = self.get_feature_vector(f)
        # saving all our feature vectors in pickled file
        with open(self.features_file_path, 'w') as fp:
            pickle.dump(result, fp)


def run():
    featureExtraction = Feature()
    featureExtraction.extract()
run()