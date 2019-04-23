## Setup
### Installation on Linux
```
sudo apt-get update
pip install --upgrade dask
```

For Cuda 8 (you need it)
```
pip install tensorflow-gpu==1.4 
```

Additional packages
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
```

Create directory for training code, model and data
```
mkdir TrafficLightClassification
cd TrafficLightClassification
```

Get models from tensorflows model repository that are compatible with tensorflow 1.4
```
git clone https://github.com/tensorflow/models.git
cd models
git checkout f7e99c0
```

Test the installation
```
cd research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```

## Training the model
### Data
#### Real World
Images with labeled traffic lights can be found on

1.  [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
2.  [LaRA Traffic Lights Recognition Dataset](http://www.lara.prd.fr/benchmarks/trafficlightsrecognition)
3.  Udacity's ROSbag file from Carla
4.  Traffic lights from Udacity's simulator

##### Using the Bosch Small Traffic Lights Dataset

The shape of Udacity's images is 1368 x 1096 (ratio 1.333) whereas the shape of the Bosch dataset is 1280 x 720 (ratio 1.777). The script `resize_tl_images` crops the Bosch's dataset to 960 x 720 to get the same ratio and produces a new yaml file.

All labels and bounding boxes are stored in a yaml file. We need a script to convert this to a TFRecord file. 

TODO: show the right path
You may find this script in https://github.com/bosch-ros-pkg/bstld/tree/master/tf_object_detection



#### Simulation
Training images for simulation can be found downloaded from Vatsal Srivastava's dataset and Alex Lechners's dataset. The images are already labeled and a  [TFRecord file](https://github.com/alex-lechner/Traffic-Light-Classification#23-create-a-tfrecord-file)  is provided as well:

1.  [Vatsal's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)
2.  [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)

### Model 

The model "SSD Mobilenet V1" was used for classification of the Bosch Small Traffic Lights Dataset. See the performance on this page https://github.com/bosch-ros-pkg/bstld .

The model "SSD Inception V2" seems to perform better at the expense of speed. See [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to see the performance comparison.

### Download
switch to the models directory and download 
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
```
extract them there
```
tar -xzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz
```

### Model Configuration

Go back to the TrafficLightClassification directory and create a config directory.

```
mkdir config
```

copy the chosen models to config
```
cp models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config config/
cp models/research/object_detection/samples/configs/ssd_inception_v2_coco.config config/
```

#### Cofiguration on Udacity Simulation dataset for "SSD Inception V2"

Configuration taken from https://github.com/bosch-ros-pkg/bstld/blob/master/tf_object_detection/configs/ssd_mobilenet_v1.config

1.  Change  `num_classes: 90`  to the number of labels in your  `label_map.pbtxt`. This will be  `num_classes: 4`
2.  Set the default  `max_detections_per_class: 100`  and  `max_total_detections: 300`  values to a lower value for example  `max_detections_per_class: 25`  and  `max_total_detections: 100`
3.  Change  `fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"`  to the directory where your downloaded model is stored e.g.:  `fine_tune_checkpoint: "models/ssd_inception_v2_coco_2018_01_28/model.ckpt"`
4.  Set  `num_steps: 200000`  down to  `num_steps: 20000`
5.  Change the  `PATH_TO_BE_CONFIGURED`  placeholders in  `input_path`  and  `label_map_path`  to your .record file(s) and  `label_map.pbtxt`

### Train
Copy `train.py` from `TrafficLightClassification/models/research/object_detection` to `TrafficLightClassification` folder

Start Training with 
```
python train.py --logtostderr --train_dir=./models/train-ssd-inception-simulation --pipeline_config_path=./config/ssd_inception_v2_coco-simulator.config
```

### Freeze
The trained model needs to be frozen for production. Just copy `export_inference_graph.py`  from `TrafficLightClassification/models/research/object_detection` to `TrafficLightClassification` folder. 

Execute:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_inception_v2_coco-simulator.config --trained_checkpoint_prefix ./models/train-ssd-inception-simulation/model.ckpt-20000 --output_directory models/frozen-ssd_inception-simulation
```

If this results in an error:
```
  File "/home/mona/src/udacity/CarND-Capstone-Root/CarND-Capstone/TrafficLightClassification/models/research/object_detection/exporter.py", line 72, in freeze_graph_with_def_protos
    optimize_tensor_layout=1)
ValueError: Protocol message RewriterConfig has no "optimize_tensor_layout" field.
```

You need to change the file `TrafficLightClassification/models/research/object_detection/exporter.py` at line 72

See Change: https://github.com/tensorflow/models/pull/3106/files

Original:
```
      rewrite_options = rewriter_config_pb2.RewriterConfig(
          optimize_tensor_layout=True)
```

Changed:
```
      rewrite_options = rewriter_config_pb2.RewriterConfig(
          layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
```

You may find the frozen graph `frozen_inference_graph.pb` in `TrafficLightClassification/models`

## Detection
The [object detection tutorial - a jupyter notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) walks you through the steps

I copy and pasted many of these steps into the detector.py

Take care that the following variables are set according to your needs:

```
MODEL_NAME = 'frozen-ssd_inception-simulation'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'udacity_label_map.pbtxt')
PATH_TO_TEST_IMAGES_DIR = 'test_images/simulation'
PATH_TO_TEST_IMAGES_OUTPUTDIR = 'test_images_results/simulation'
```

execute 
```
python detector.py
```
The resulting images can be found in the directory `PATH_TO_TEST_IMAGES_OUTPUTDIR`