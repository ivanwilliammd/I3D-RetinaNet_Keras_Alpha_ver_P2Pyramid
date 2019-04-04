"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
import cv2
import h5py
from PIL import Image

from .transform import change_transform_origin


def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    hdf5_file=h5py.File(path, 'r')
    dataset_hdf5=hdf5_file.get('dataset')
    image=np.array(dataset_hdf5)
    # image = np.asarray(Image.open(path).convert('RGB'))
    
    ## Change from image input to HDF5 input
    ## import IPython;IPython.embed()
    return image.copy()


def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    depth, height, width, channels = image.shape

    # print('debug adjust_transform_for_image --> cek depth, height, width, channels')
    # import IPython;IPython.embed()

    result = transform

    # print('debug adjust_transform_for_image --> cek result sebelum relative_translation')
    # import IPython;IPython.embed()

    # Scale the translation with the image size if specified.
    if relative_translation:
        # result[0:2, 2] *= [width, height]
        result[0:2, 2] *= [width, height]
    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result
    # # print('debug adjust_transform_for_image --> cek result SEBELUM change_transform_origin')
    # # import IPython;IPython.embed()

    # result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    # print('debug adjust_transform_for_image --> cek result SETELAH change_transform_origin')
    # import IPython;IPython.embed()

    # result_array=np.stack((result, result, result, result, result, result, result, result,
    #     result, result, result, result, result, result, result, result,
    #     result, result, result, result, result, result, result, result,
    #     result, result, result, result, result, result, result, result), axis=0)

    # return result, result_array


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    # output = cv2.warpAffine(
    #     image,
    #     matrix[:2, :],
    #     dsize       = (image.shape[1], image.shape[0]),
    #     flags       = params.cvInterpolation(),
    #     borderMode  = params.cvBorderMode(),
    #     borderValue = params.cval,
    # )
    # return output

    ## GANTI OUTPUT OPEN CV2
    # import IPython;IPython.embed()
    i=0
    output_list=[]
    output_array=[]
    # print('DEBUG: cek image dan matrix')
    # import IPython;IPython.embed()

    for i in range(len(image)):
        image_translation=image[i]
        output = cv2.warpAffine(
            image_translation, matrix[:2, :],
            dsize       = (image_translation.shape[1], image_translation.shape[0]),
            flags       = params.cvInterpolation(),
            borderMode  = params.cvBorderMode(),
            borderValue = params.cval,
        )        
        # output = cv2.warpAffine(
        #     image,
        #     matrix[:2, :],
        #     dsize       = (image[i].shape[1], image[i].shape[0]),
        #     flags       = params.cvInterpolation(),
        #     borderMode  = params.cvBorderMode(),
        #     borderValue = params.cval,
        # )
        output_list.append(output)
        # i=i+1
        
    output_array=np.stack((output_list[0], output_list[1], output_list[2], output_list[3], output_list[4], 
        output_list[5], output_list[6], output_list[7], output_list[8], output_list[9],
        output_list[10], output_list[11], output_list[12], output_list[13], output_list[14], 
        output_list[15], output_list[16], output_list[17], output_list[18], output_list[19],
        output_list[20], output_list[21], output_list[22], output_list[23], output_list[24], 
        output_list[25], output_list[26], output_list[27], output_list[28], output_list[29],
        output_list[30], output_list[31]), axis=0)
    # import IPython;IPython.embed()
    
    return output_array


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # # compute scale to resize the image
    # scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # # resize the image with the computed scale
    # img = cv2.resize(img, None, fx=scale, fy=scale)

    # return img, scale

    
    
    i=0
    img_list=[]
    img_array=[]

    for i in range(len(img)):
        # # resize the image with the computed scale
        scale = compute_resize_scale(img[i].shape, min_side=min_side, max_side=max_side)
        img[i] = cv2.resize(img[i], None, fx=scale, fy=scale)    
        img_list.append(img[i])
        
            
    img_array=np.stack((img_list[0], img_list[1], img_list[2], img_list[3], img_list[4], img_list[5], img_list[6], img_list[7], img_list[8], img_list[9],
        img_list[10], img_list[11], img_list[12], img_list[13], img_list[14], img_list[15], img_list[16], img_list[17], img_list[18], img_list[19],
        img_list[20], img_list[21], img_list[22], img_list[23], img_list[24], img_list[25], img_list[26], img_list[27], img_list[28], img_list[29],
        img_list[30], img_list[31]), axis=0)
    return img_array, scale