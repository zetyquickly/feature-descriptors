import cv2 as cv
import numpy as np

def detector_factory(cfg):
    type = cfg.model.detector_type
    if type == "SIFT":
        detector = cv.xfeatures2d.SIFT_create()
    elif type == "SURF":
        detector = cv.xfeatures2d.SURF_create()
    elif type == "KAZE":
        detector = cv.KAZE_create()
    elif type == "AKAZE":
        detector = cv.AKAZE_create()
    elif type == "BRISK":
        detector = cv.BRISK_create()
    elif type == "ORB":
        detector = cv.ORB_create()
    else:
        raise NotImplementedError(f"Detector of type {type} is not supported")

    return detector

def descriptor_factory(cfg):
    type = cfg.model.descriptor_type
    if type == "SIFT":
        descriptor = cv.xfeatures2d.SIFT_create()
    elif type == "SURF":
        descriptor = cv.xfeatures2d.SURF_create()
    elif type == "KAZE":
        descriptor = cv.KAZE_create()
    elif type == "AKAZE":
        descriptor = cv.AKAZE_create()
    elif type == "BRISK":
        descriptor = cv.BRISK_create()
    elif type == "ORB":
        descriptor = cv.ORB_create()
    elif type == "BRIEF":
        descriptor = cv.xfeatures2d.BriefDescriptorExtractor_create()
    elif type == "FREAK":
        descriptor = cv.xfeatures2d.FREAK_create()
    else:
        raise NotImplementedError(f"Descriptor of type {type} is not supported")

    return descriptor

def matcher_factory(cfg):
    matcher_type = cfg.model.matcher_type
    descriptor_type = cfg.model.descriptor_type
    if matcher_type == "BF":
        def matcher(queryDescriptors, trainDescriptors):
            if (descriptor_type == 'SIFT' or descriptor_type == 'SURF' or 
                    descriptor_type == 'KAZE'):
                normType = cv.NORM_L2
            else:
                normType = cv.NORM_HAMMING
            BFMatcher = cv.BFMatcher(normType = normType,
                                     crossCheck = True)
            matches = BFMatcher.match(
                queryDescriptors = queryDescriptors,
                trainDescriptors = trainDescriptors
            )
            matches = sorted(matches, key = lambda x: x.distance)
            return matches

    elif matcher_type == "FLANN":
        def matcher(queryDescriptors, trainDescriptors):
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                                trees = 5)
            search_params = dict(checks = 50)
            queryDescriptors = queryDescriptors.astype(np.float32)
            trainDescriptors = trainDescriptors.astype(np.float32)

            FLANN = cv.FlannBasedMatcher(
                indexParams = index_params,
                searchParams = search_params
            )

            matches = FLANN.knnMatch(
                queryDescriptors = queryDescriptors,
                trainDescriptors = trainDescriptors,
                k = 2
            )
            ratio_thresh = 0.7
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            return good_matches
    else:
        raise NotImplementedError(f"Matcher of type {descriptor_type} is not supported")

    return matcher


def feature_extractor_factory(detector, descriptor):
    def extractor(image):
        keypoints = detector.detect(image, None)
        keypoints, descriptors = descriptor.compute(image, keypoints)
        return keypoints, descriptors
    return extractor