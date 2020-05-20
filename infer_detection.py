import cv2
import numpy as np
import tensorflow as tf

import sys
import argparse
import os
import glob

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'frozen_inference_graph.pb'
IMAGE_NAME = 'IoT_And_Smart_Buildings.png'#'UF2013_ps-cn-ist-1_0029_001-1.jpg'

# Get the working folder path
CWD_PATH = os.getcwd()

PATH_TO_LABELS = os.path.join(CWD_PATH, 'annotations', 'label_map.pbtxt')

#In this case 3 persons 
NUM_CLASSES = 3
print('Labels at: {}'.format(PATH_TO_LABELS))

def infer_fn(path_to_image, path_to_ckpt, path_to_output):
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes = NUM_CLASSES, use_display_name = True)
        category_index = label_map_util.create_category_index(categories)

        # Init a tf graph and extract the model to it
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph = detection_graph)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Get the images in the folder
        test_img_paths = glob.glob(os.path.join(path_to_image, '*.png'))
        print(test_img_paths)
        for path in test_img_paths:
                file_name = path.split('/')[-1]
                print('\nFile name: {}'.format(file_name))
                # Load image using OpenCV and
                # expand image dimensions to have shape: [1, None, None, 3]
                # i.e. a single-column array, where each item in the column has the pixel RGB value
                image = cv2.imread(path)
                image_expanded = np.expand_dims(image, axis = 0)
                print('Image size: {}\n'.format(image_expanded.shape))
                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict ={image_tensor: image_expanded})
                print(np.squeeze(scores)[:4])
                vis_util.visualize_boxes_and_labels_on_image_array(
                            image,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=2,
                            min_score_thresh=0.4
                        )
                output_file = os.path.join(path_to_output, file_name)
                cv2.imwrite(output_file, image)

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_image_folder', default='', type=str)
        parser.add_argument('--output_image_folder', default='', type=str)
        parser.add_argument('--graph_folder', default='', type=str,
                help='Folder locating the inference graph')
        args = parser.parse_args()

        path_to_ckpt = os.path.join(CWD_PATH, args.graph_folder, MODEL_NAME)
        path_to_image = os.path.join(CWD_PATH, args.test_image_folder)
        path_to_output = os.path.join(CWD_PATH, args.output_image_folder)
        print('Images at: {}'.format(path_to_image))
        print('Checkpoints at: {}'.format(path_to_ckpt))

        infer_fn(path_to_image, path_to_ckpt, path_to_output)

