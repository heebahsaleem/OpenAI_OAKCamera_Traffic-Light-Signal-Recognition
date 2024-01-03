from src import config
from src import utils
import depthai as dai
import cv2
import tensorflow as tf
import numpy as np
import os
print("depth image pipeline.....")
pipeline = utils.create_pipeline_images()

# pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # define the queues that will be used in order to communicate with
    # depthai and then send our input image for predictions
    classifierIN = device.getInputQueue("classifier_in")
    classifierNN = device.getOutputQueue("classifier_nn")
    print("[INFO] loading image from disk...")
    for img_path in config.TEST_DATA:
        # load the input image and then resize it
        image = cv2.imread(img_path)
        image_copy = image.copy()
        nn_data = dai.NNData()
        nn_data.setLayer(
            "input",
            utils.to_planar(image_copy, config.IMG_DIM)
        )
        classifierIN.send(nn_data)
        print("[INFO] fetching neural network output for {}".
            format(img_path.split('/')[1]))
        in_nn = classifierNN.get()
        # apply softmax on predictions and
        # fetch class label and confidence score
        if in_nn is not None:
            data = utils.softmax(in_nn.getFirstLayerFp16())
            result_conf = np.max(data)
            if result_conf > 0.5:
                result = {
                    "name": config.LABELS[np.argmax(data)],
                    "conf": round(100 * result_conf, 2)
                }
            else:
                result = None
         # if the prediction is available,
        # annotate frame with prediction and show the frame
        if result is not None:
            cv2.putText(
                image,
                "{}".format(result["name"]),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                image,
                "Conf: {}%".format(result["conf"]),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.imwrite(
                config.OUTPUT_IMAGES +"/"+img_path.split('/')[1],
                image
            )
