# I3DR-Net Alpha Stage using Keras - Added P2 Layer with Original RetinaNet anchors [DROPPED]

Adapted from [keras-retinanet](https://github.com/fizyr/keras-retinanet) and [Kinetics I3D](https://github.com/dlpbc/keras-kinetics-i3d/releases) for lung nodule detection. Please cite original code & publication when using this repo.

In our experiment, adding P2 layer will make OOM on Tesla P100 16GB, so use it with caution.

## Check out other repo:
1. [I3DR-Net Original without weight, and original anchors](https://github.com/ivanwilliammd/I3D-RetinaNet_Keras_Alpha_ver_LargeObject)
2. [I3DR-Net Upsampled with P2 pyramid](https://github.com/ivanwilliammd/I3D-RetinaNet_Keras_Alpha_ver_P2Pyramid)
3. [I3DR-Net Original with smaller anchors](https://github.com/ivanwilliammd/I3D-RetinaNet_Keras_Alpha_ver_SmallObject)
4. [JPG to HDF5 Converter](https://github.com/ivanwilliammd/BatchImagesToHDF5_Converter)
5. [I3DR-Net-Transfer-Learning](https://github.com/ivanwilliammd/I3DR-Net-Transfer-Learning)

------------------------------------------------------------------------------
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

```http://www.apache.org/licenses/LICENSE-2.0```

------------------------------------------------------------------------------
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************
