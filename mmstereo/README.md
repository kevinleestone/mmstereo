# MMStereo

Learned stereo training code for the paper:

```
Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, and Max Bajracharya. A Learned Stereo Depth System for Robotic Manipulation in Homes. ICRA 2022 submission.
```

## Setup

### Install apt dependencies
```
sudo apt install libturbojpeg virtualenv python3.8
```

### Python virtual environment
`virtualenv` lets you install Python packages without conflicting with other workspaces.

#### Create the virtual environment
```
virtualenv -p python3.8 venv
```

#### Activate the virtual environment
```
source venv/bin/activate
```
Make sure you activate it each time you use the training code.

### Install PyTorch
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install other dependencies
```
pip install -r requirements.txt
```

## Sceneflow Dataset Training

### Get dataset
Download RGB and disparity images for Sceneflow Flying Things dataset from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html.

Extract the downloaded files to `datasets/sceneflow/raw`.


### Prepare dataset
The dataset needs to be converted into a format that is compatible with the training code.

```
python prepare_sceneflow.py
```

### Train
The configuration is set up to work with a Titan RTX GPU with 24GB of VRAM. For other GPUs with less VRAM, you may need to reduce the batch size.
```
python train.py --config config_sceneflow.yaml
```

### Visualize with tensorboard
In a new terminal (since your other terminal is running the training script),
```
source venv/bin/activate
tensorboard --logdir output
```
and then open the url it gives you in a browser.

# Inference
You can run inference on a pair of images using the Torchscript output from training.
```
python run.py --script output/sceneflow/version_0/checkpoints/model.pt --left datasets/sceneflow/flying_things/val/left/0000000.png --right datasets/sceneflow/flying_things/val/right/0000000.png
```

# Dataset
Each sample is dataset is made up of a pair of left and right RGB images, a left disparity image, and an optional right
disparity image.

## RGB images
The RGB images should be 3-channel 8-bit images as either JPG or PNG format.

## Disparity images
The disparity images should a floating point Numpy ndarray stored in NPZ format with the same height and width as the
RGB images. If there is no right disparity, training is possible but the horizontal flip data augmentation can't be
used.

## Directory structure
The each element of a sample should be stored in a certain directory with the same base filename and appropriate
extension.

Example structure:
* left
  * sample0.png
  * sample1.png
* left_disparity
  * sample0.npz
  * sample1.npz
* right
  * sample0.png
  * sample1.png
