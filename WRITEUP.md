# Project Write-Up

## Explaining Custom Layers

Any layer that is not in the list of currently supported layers is automatically classified as a custom layer. Handling this kind of layers is very important because regularly new advances are being made in deep learning that involve the use of new layers that didn't existed previously, and that therefore OpenVINO don't know how to optimize properly.

To add custom layers there are a few differences depending on the framework in which the original model was created. In both TensorFlow and Caffe we can register the custom layers as extensions to the Model Optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer (we need Cafee for doing this).

For TensorFlow, its second option is to replace the unsupported subgraph with a different subgraph. As a last resort we could offload the 'unknown computations' to tensorflow directly, which obvioisly will result in some loss of inference speed.

## Comparing Model Performance

My method(s) to compare models over several different metrics were the following:
* I examined the outputs of the app while trying different confidence thresholds and analyzing the inference speed visually by comparing the velocity at which the annotated video was being reproduced in the app for the different models. This could have been performed formaly with a couple of `time.time()` statements but given the huge differences that could be perceived without any calculation i didn't considered it truly necessary. What i found is that:
    * Without any doubt the model that produced the fastest inference was the model download from the OpenVINO Model Zoo. Closely behind was the Mobilenet model. On the other hand the Inception model inference time was easily around 2 or 3 times slower than the other models. This agrees with the relative differences in speed that were mentioned in the [Tensorflow 1 Model Zoo Github page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
    * The High angle pedestrian detector model was the only one that could succesfully detect the second person almost in all the frames that he appeared. The models that i converted manually had very serious problems with this 'problematic person', but worked well with all the remaining persons (feel free to try them out).

* I compared the filesize of the Intermediate Representation .bin files that basically contain all the weights of the models. The results show that the model downloaded form the OpenVINO Model Zoo has a hugely smaller filesize than the other models. The file sizes were the following:
    * Resnet model: 111 Mb.
    * Inception model: 95.4 Mb.
    * MobileNet model: 64 Mb.
    * High angle pedestrian detector (OpenVINO Model Zoo): 2.8 Mb!! (wow).
    
* I compared the file size of the uncompressed models downloaded from the Tensorflow 1 Model Zoo (frozen_inference_graph.pb) vs the converted Intermediate Representation (.bin files). In general we achieved approximately a 6/7 % decrease in the model size:
    * Mobilenet: 68 Mb (pre) vs 64 Mb (post).
    * Inception: 99.5 (pre) vs 95.4 Mb (post).
    * Resnet: 118 Mb (pre) vs 111 Mb (post).

* As a general reminder, one of the most important advantages of AI on the edge is that we don't need a network connection,that principaly the only costs of this approach are the cost of the devices and their power consumption, and that there are no confidientality issues because all the inferences are being peformed locally (in contrast to cloud where we pay as long as we use a service and where we need to stream our inputs to the internet for processing and producing inferences).

## Assess Model Use Cases

Some of the potential use cases of the people counter app are the following:

* Safety: Ensure that no more than a specified number of people are in sensitive or critical places (for example to avoid that people obstruct the exits of buildings in case of fires or earthquakes).

* Shopping: Tracking the number of people that enter into the different stores of a shopping mall to asses popularity and highly concurred regions and, for example, adjusting the rent prices of the different shops dynamically.

* Marketing: Tracking how many people stand in front of several key regions (like ads, promotions and product showcases) for assisng the effectivity of new marketing approaches, product launches, museums and expositions, etc...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. Regarding model accuracy, there is often a tradeoff between inference speed/model size and inference accuracy. The desired specifications of the system will always depend on what is the problem under study. 

Several environmental conditions (like lightning) have to be either controlled or appropiate training examples have to be used for training (so that our models are more robust to these kind of situations). Image augmentation techinques are  usually very helpful for this task. 

Image size in general will always be different depending on the specification of the cameras or the inputs that we use. We need to preprocess this images to account for different posible ordering of color channels, reshaping the inputs to match the expected model input size,  and sometimes removing noise or applying some transformations on the models inputs like pixels value normalization or additional techniques like edge detection (if applicable).

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [SSD Mobilenet V2 COCO](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - Source: [Tensorflow 1 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
  - To convert the model i first used the `wget` command to download the model into the workspace, `tar -xvf` to uncompress the model, `cd` to change into the newly created model directory and finally ran the Model Optimizer script to create the Intermediate Representation with the following command:
  
   `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  
  - The model was insufficient for the app because even with confidence thresholds as low as 0.3 the second person (men with a black jacket) couldn't be recognized in several frames (sometimes he spent around 5 seconds without being detected). Therefore, i tried to test additional models.
  - I tried to improve the model for the app by trying several confidence thresholds and adding some extra code to give the detection some 'relaxation time' for avoiding counting the same person as different persons (with this code the app needs to see no detections for a specified number of frames before it will consider a new detection to be a different person).
  
- Model 2: [SSD Inception V2 COCO](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - Source: [Tensorflow 1 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
  - To convert the model i first used the `wget` command to download the model into the workspace, `tar -xvf` to uncompress the model, `cd` to change into the newly created model directory and finally ran the Model Optimizer script to create the Intermediate Representation with the following command:
  
   ` python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  - The model was insufficient for the app for the same reasons as before.
  - I tried to improve the model for the app with the same methods as the past model.

- Model 3: [Faster RCNN Resnet50 COCO](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)
  - Source: [Tensorflow 1 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
  - To convert the model i first used the `wget` command to download the model into the workspace, `tar -xvf` to uncompress the model, `cd` to change into the newly created model directory and finally ran the Model Optimizer script to create the Intermediate Representation with the following command:
  
   `  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --output=detection_boxes,detection_scores,num_detections`
   
  - The model was insufficient for the app because i couldn't even get the model to work with the already tested app ( the IR was generated successfully). When trying to use the model with the already tested app no output was being generated. I tried removing the FFMPEG part of the code to be able to print intermediate variables and debug the code and i found out that the input shape of the model was sometimes randomly being [1,3] and other times it was [1,3,600,600]. In neither of these cases the model was able to produce any output, and when running the app without the FFMPEG part it even produced some unfortunate segmentation fault errors (the app was already tested and i just changed the model that was fed into the app).
  
Because none of this models was good enough for the application i **ended up using the folowing pretrained model in the Open Model Zoo: [High angle pedestrian detection model](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html).**

After configuring all the corresponding FFMPEG, UI and MQTT servers, we can run the app using this model with the following command:

`python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

