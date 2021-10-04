import cv2 as cv
import os
import shutil
import sys


import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import factories
from utils import image_resize

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg):
    # get detector, descriptor, extractor and matcher using factory methods
    try:
        detector = factories.detector_factory(cfg)
        descriptor = factories.descriptor_factory(cfg)
        extractor = factories.feature_extractor_factory(detector, descriptor)
        matcher = factories.matcher_factory(cfg)
    except Exception as e:
        logging.critical(e, exc_info=True)
        logging.info(f"Failed on model initialization")
        return None

    # load query image
    path = os.path.join(
        cfg.data.query_image_path, 
        os.listdir(cfg.data.query_image_path)[0]
    )
    query_image = cv.imread(filename = path,
                            flags = cv.IMREAD_GRAYSCALE)
    query_image = image_resize(query_image, width=cfg.data.image_target_width)
    try:
        query_keypoints, query_descriptors = extractor(query_image)
    except Exception as e:
        logging.critical(e, exc_info=True)
        logging.info(f"Broke on query image feature extraction")
        return None

    # check output directory
    if not os.path.exists(cfg.data.output_dir_path):
        os.mkdir(cfg.data.output_dir_path)
    # for each image in training set directory find matches and draw results
    for item in os.listdir(cfg.data.train_set_path):
        train_image_path = os.path.join(cfg.data.train_set_path, item)
        train_image = cv.imread(filename = train_image_path,
                                flags = cv.IMREAD_GRAYSCALE)                      
        train_image = image_resize(train_image, width=cfg.data.image_target_width)
        try:
            train_keypoints, train_descriptors = extractor(train_image)
        except Exception as e:
            logging.critical(e, exc_info=True)
            logging.info(f"Broke on {train_image_path} image feature extraction")
            return None

        try:
            matches = matcher(query_descriptors, train_descriptors)
            logging.info(f"Found {len(matches)} matches between query and {item}")
        except Exception as e:
            logging.critical(e, exc_info=True)
            logging.info(f"Broke on {train_image_path} image feature matching")
            return None

        output = cv.drawMatches(img1 = query_image,
                                keypoints1 = query_keypoints,
                                img2 = train_image,
                                keypoints2 = train_keypoints,
                                matches1to2 = matches[:cfg.data.num_matches_show],
                                outImg = None,
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(os.path.join(cfg.data.output_dir_path, item), output)
    logging.info("Execution succeeded!")

if __name__ == "__main__":
    my_app()  

# run multirun with
# python my_app.py --multirun model.detector_type=SIFT,SURF,KAZE,AKAZE,ORB,BRISK model.descriptor_type=SIFT,SURF,KAZE,AKAZE,ORB,BRISK,BRIEF,FREAK model.matcher_type=BF,FLANN