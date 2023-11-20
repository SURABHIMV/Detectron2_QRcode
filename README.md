# QR code detection using the Detectron2 model

This repository contains code to detect QR codes in an image. It also mentions the label ('QRCODE') and the percentage, indicating how much percentage of the QR code is detected, using a finetuned Detectron2 model.

## Files

* `Detectron2.ipynb`: This notebook contains code about how to fine-tune the Detectron2 model with QR code data and detect the QR code in an image. Given that the JSON file in both train and test files does not contain segmentation information, the 'segmentation' field in the JSON file is updated with rectangular segmentation based on bounding box information. Once the data is prepared according to the model requirements, fine-tune a COCO-pre-trained R50-FPN Mask R-CNN model on the QRcode dataset. It takes around 2 minutes to train 300 iterations on a P100 GPU. Finally, the results are visualized using the test data.
  
* `QR Code.v2i.coco.zip`: After unzipping this file contains two files train and a test file. train file contains annotations.coco.json (which contains information about the images and their annotation part) and train QRcode images. similarly in test file contains a JSON file and test QRcode images.

## Libraries and Versions

The following libraries and versions were used in this project:

* `pyyaml==5.1`
* `torch==1.11`
* `cuda==cu113`
* `detectron2==0.6`
* `numpy==1.24.3`
