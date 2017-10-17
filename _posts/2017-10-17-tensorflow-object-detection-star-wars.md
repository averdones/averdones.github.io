---
published: false
---

In this post, I'm going to train an object detector to locate R2-D2 and BB-8 in an image or video. But let's not wait and see some results!

<div class="imgcap">
<img src="/images/tensorflow-object-detection-star-wars/result_1.gif">
<div class="thecap">Detection of R2-D2 and BB-8</div>
</div>

## What is Transfer Learning

Training an entire Convolutional Neural Network (ConvNet or CNN) from scratch is something that few people can afford. In order to accomplish such task, two elements are required in large amounts: **data** and **computational power**. The former is hard to get: huge and reliable datasets are normally the result of a joint effort between different people or research groups working together for an extended period of time. The latter could be very expensive (in terms of money or time) if a huge model needs to be trained.

For these reasons, what many people do is what is called **transfer learning**: use an already trained (also called pre-trained) network as the initialization point for their own model, or more specifically, use the weights of the pre-trained ConvNet as the starting weights of their own model. At this point, two options exist:
- **Leave the net as it is**, perhaps modifying and re-training only the last layer (the fully connected one). This is the easiest case where we can start making predictions right away (if we don't even re-train the last layer). The [demo](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) in the Tensorflow repo covers this situation and it can be set up in a few minutes.
- **Fine-tune the net**. In this case, we re-train the weights of the ConvNet using regular *backpropagation*. Since the initial layers of a CNN tend to explain more primitive shapes, common to many different objects, we could choose to re-train only the higher layers and fix the lower ones (the ones closer to the input). For the same reason, this method is convenient if we want to detect different objects from the ones used to train the pre-trained ConvNet. Here, I will show you this process in order to detect R2-D2 and BB-8 from Star Wars in any image or video.

If you're asking yourself *"Why do you want to detect R2-D2 and BB-8 in a video?"*, I guess the easiest answer would be *"Why not?"*. The experiment at hand is an engineering one or more colloquially a problem of *How to put the pieces together*. I wanted to try Tensorflow's Object Detection API and make it work. My objective was not to achieve state-of-the-art scores. Therefore, I figured I'd use something cool that I like. So... Star Wars! Of course, you can collect your own images and detect whatever object you want.

## Tensorflow's Object Detection API 

Some time ago, the Tensorflow team made available an Object Detection API that makes the process of fine-tuning a pre-trained model easier. Without it, you should manually download the weights of a pre-trained neural network and write all the Tensorflow code (or other) to build a net and initialize it with the pre-trained weights. In order to use the API, we only need to tweak some lines of code from the files already made available to us. Here, I won't go into the details of the architecture of the net, the optimization algorithm used (the default will be RMSProp) or the several other hyper-parameters.

Following, I will list the main steps needed to build your own object detection model:

1. **Collect your images** and **annotate** them (I used [labelImg](https://github.com/tzutalin/labelImg)). Normally, you would want to have a few hundred images but it depends on the problem. I managed to gather a hundred for each class (R2-D2 and BB-8). They are probably too few and too similar based on the results obtained, but at least the model works as expected.
2. **Create TF Records** using the [file provided](https://github.com/averdones/star_wars_object_detection/blob/master/create_sw_tf_record.py) by the API. These files will be the input for the API. If your dataset (images and annotations) has an analogous format as the one of the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/), you shouldn't have any trouble creating your TF Records. Tensorflow has also a [helper code](https://github.com/tensorflow/models/blob/master/research/object_detection/create_pascal_tf_record.py) to create TF Records of the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). Otherwise, this step could be a bit tricky, since you should code the program by yourself. The command to run the script that creates the TF Records if the following:

```
python create_sw_tf_record.py --
label_map_path=D:/Python_projects/Detection/object_detection/data/sw_label_map.pbtxt --
data_dir=D:/Python_projects/Detection --output_dir=D:/Python_projects/Detection/object_detection/data
```










