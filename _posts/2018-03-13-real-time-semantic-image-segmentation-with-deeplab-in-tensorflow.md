---
layout: post
title: >-
  Real-time semantic image segmentation with DeepLab in Tensorflow
comments: true
published: true
---

A couple of hours ago, I came across the [new blog of Google Research](https://research.googleblog.com/2018/03/semantic-image-segmentation-with.html). This time the topic addressed was **Semantic Segmentation** in images, a task of the field of Computer Vision that consists in assigning a semantic label to every pixel in an image. You can refer to [the paper](https://arxiv.org/abs/1802.02611) for an in-depth explanation of the new version of the algorithm they used (DeepLab-v3+).  

Semantic segmentation is a more advanced technique compared to *image classification*, where an image contains a single object that needs to be classified into some category, and *object detection and recognition*, where an arbitrary number of objects can be present in an image and the objective is to detect their position in the image (with a bounding box) and to classify them into different categories. 

The problem of semantic segmentation can be thought as a much harder object detection and classification task, where the bounding box won't be a box anymore, but instead will be an irregular shape that should overlap with the real shape of the object being detected. Detecting each pixel of the objects in an image is a very useful method that is fundamental for many applications such as autonomous cars.

In this post, I will share some code so you can play around with the latest version of DeepLab (DeepLab-v3+) using your webcam in real time. All my code is based on the excellent [code published](https://github.com/tensorflow/models/tree/master/research/deeplab) by the authors of the paper. I will also share the same notebook of the authors but for Python 3 (the original is for Python 2), so you can save time in case you don't have tensorflow and all the dependencies installed in Python 2.

But first, a quick example of what I'm talking about:

<img src="/assets/images/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/webcam_segmentation.gif" class="center">

*P.S. Don't worry, I'm not choking, I just forgot to change the sneaky BGR in OpenCV to RGB.*

In order to run my code, you just need to follow the instructions found in the [github page](https://github.com/tensorflow/models/tree/master/research/deeplab) of the project, where the authors already prepared an [off-the-shelf jupyter notebook](https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb) to run the algorithm on images. I only use an extra dependency which is OpenCV. And optionally, [scikit video](http://www.scikit-video.org/stable/), in case you also want to save the video. 

Copy the following snippet into a jupyter notebook cell that should be inside the directory of deeplab (that you previously should've cloned) and just run it! Now you can see yourself and a real-time segmentation of everything captured by your webcam (of course, only the objects that the net was trained on will be segmented).

If you get an error, you probably need to change the line that shows `final = np.zeros((1, 384, 1026, 3))` based on your camera resolution. Here, the shape of `color_and_mask` is needed.

Every time you run the code, a new model of approximately 350Mb will be downloaded. So, if you want, you can just change the line where it says `model = DeepLabModel(download_path)` to a local path where you stored your downloaded model.

This is the code to run DeepLab-v3+ on your webcam:
{% gist averdones/05ac8d828ab7c2daa508af43f520d8e4 %}

And this is the code to run DeepLab-v3+ on images using Python 3:
{% gist averdones/d9b66a8078391ebc18d17bdadeca7dce %}

Have fun segmenting!



References:

*@article{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  journal={arXiv:1802.02611},
  year={2018}
}*