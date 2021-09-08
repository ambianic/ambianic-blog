# Comparing MoveNet to PoseNet for Person Fall Detection

_by [Bhavika Panara](https://github.com/bhavikapanara)_

MoveNet and PoseNet are computer vision models for Pose Estimation. Their architecture is composed of several layers. First, they detect a human figure in an image and then estimate spatial locations of key body joints (key points), such as someone’s elbow, shoulder or foot showing up in an image.

It is important to be aware of the fact that pose estimation merely estimates where key body joints are and does not recognize who is in an image or video.

## Applications

There are two popular classes of applications for pose estimation: Healthcare and Fitness/yoga.

### Health Applications

One popular class of applications for pose detection is physical therapy. Monitoring movement and posture for deviations from established medical norms and facilitating proactive guidance to patients.

During the COVID19 pandemic when elderly people have been isolated and required remote monitoring, pose estimation has helped with remote fall detection and alerting in real-time in systems such as Ambianic.ai. Ambianic Fall Detector is a Raspberry Pi 4B based smart camera that uses pose estimation to continuously monitor high-risk areas of a home for possible falls and instantly alerts caregivers and family when a loved one has fallen down. 

<img src="https://user-images.githubusercontent.com/2234901/131890805-d15b647d-0072-422c-b98d-f80e7f83833b.png" width=300/>

_A screenshot from Ambianic Fall Detector_

### Fitness/yoga

Pose Estimation allows detecting athletic movements such as yoga, weight lifting, squats etc. Pose estimation models allow us to track joint positions such as shoulders, elbows, hips in real-time. These fitness routines can be built digitally which are prescribed by therapists.

## Pose Estimation Model Architectures
Pose estimation allows us to build an interactive fitness application that guides users in their fitness program with the comfort of their own homes. This service can be run web-based or locally that delivers precise key points. It helps users to count their exercises and keep their historical records.

![image10](https://user-images.githubusercontent.com/2234901/131891163-7a5b50f6-f7a0-4d5d-ab77-4af26421924b.png)

_PoseNet image from [Tensorflow Blog](https://blog.tensorflow.org/2018/07/move-mirror-ai-experiment-with-pose-estimation-tensorflow-js.html)_

The Google AI Tensorflow team introduced various pose estimation models in the past couple of years with a variety of model architectures: Posenet, MoveNet Model and Blazepose. All these models have various variants of model architectures. 

Blazepose model is offered by MediaPipe and it infers 33 key points of a human body (in [2D space](https://blog.tensorflow.org/2021/05/high-fidelity-pose-tracking-with-mediapipe-blazepose-and-tfjs.html) and [3D space](https://blog.tensorflow.org/2021/08/3d-pose-detection-with-mediapipe-blazepose-ghum-tfjs.html) versions) where [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) and [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) infer 17 key points in 2D space. MoveNet is considered the new generation version of PoseNet. All 3 models are available on [Tensorflow Hub](https://tfhub.dev/) in several runtime formats such as Tensorflow Javascript (TF.js), TFLite and Coral (Edge TPU). 

TF.js model format allows us to deploy ML models in JavaScript and use ML directly in the browser or in Node.js.

TFLite model format allows us to deploy ML models in mobile and IoT devices and run on-device inference. TensorFlow Lite is a lightweight version (around 1MB binary vs >1GB for a full Tensorflow install) of TensorFlow designed for mobile and embedded devices. TensorFlow Lite models are 5-10x compressed versions of full TF models. TFLite models usually measure in 10s of MB vs 100s of MB for the original models. The compression is done  via techniques such as [quantization](https://www.tensorflow.org/lite/performance/model_optimization), which converts 32-bit parameter data into 8-bit representations.

Until recently it was not possible to train a model directly with TensorFlow Lite. We had to first train a model with TensorFlow/Keras, then save the trained model,  convert it to a  TFLite model using [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert/) and then deploy it on an edge device.  

In 2020  the TensorFlow team introduced [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) that enables us to train certain types of models on-device with custom datasets. It uses a transfer learning approach to reduce the required amount of training data and shorten the training time.

Tensorflow Lite enables us to deploy models on devices with CPU only as well as devices with support for Edge TPU. Edge TPU provides a specific set of neural network operations and architectures and it is capable of executing deep neural networks about 10x faster than a CPU. It supports only TensorFlow Lite models that are fully 8-bit quantized and then compiled specifically for the Edge TPU.

![image5](https://user-images.githubusercontent.com/2234901/131892042-29e3be63-6638-4bc3-bcb6-85ae530cec02.png)

_Converting a regular Tensorflow model to TFLite and optionally to Edge TPU for embedded device deployment_

**PoseNet:**

PoseNet is an older generation pose estimation model released in 2017. It is trained on a standard COCO dataset and provides a single pose and multiple pose estimation variants. The single pose variant can detect only one person in an image/video and the multi pose variant can detect multiple persons in an image/video. Both variants have their own set of parameters and methodology. Single pose estimation is simpler and faster but required to have a single person in an image/video otherwise key points from multiple persons will likely be estimated as being part of a single subject.

PoseNet again has two variants in terms of model architecture that is MobileNet v1 architecture and ResNet50 architecture. The MobileNetV1 architecture model is smaller and faster but has lower accuracy. The ResNet50 variant is larger and slower but it's more accurate. Both MobileNetV1 and ResNet50 variants support single pose and multi-person pose estimation.

The model returns the coordinates of the 17 key points along with a confidence score.

**MoveNet:**

MoveNet is the latest generation pose estimation model released in 2021. MoveNet is an ultra-fast and accurate model that detects 17 key points of a body. MoveNet has two variants known as Lightning and Thunder. Lightning is meant for latency-critical applications, while Thunder is meant for applications that require high accuracy. Both variants support 30+ FPS on most modern desktops, laptops, and phones. MoveNet outperforms PoseNet on a variety of datasets.

MoveNet is trained on two datasets, COCO and an internal Google dataset called Active. Active was produced by labelling key points on yoga, fitness, and dance videos from YouTube. This dataset is built by selecting three frames from each video for training. Evaluations on the Active validation dataset show a significant performance.  The initial version of MoveNet only supported single pose estimation, however a multi-person tracking version is [under development](https://twitter.com/TensorFlow/status/1427721932493963267).

MoveNet is a bottom-up estimation model. The architecture consists of two components: a feature extractor and a set of prediction heads. The feature extractor in MoveNet is MobileNetV2.

There are four prediction heads attached to the feature extractor namely Person centre heatmap( geometric centre of person), Keypoint regression(predicts a full set of key points for a person), Person keypoint heatmap(predicts the location of all key points), 2D per-keypoint offset field. These predictions are computed in parallel. 

## Fall-Detection challenges in real-world settings

### Base Heuristic Algorithm
One of the promising use cases of the pose estimation model is to detect human falls. By analyzing frame sequences using pose estimation, we can predict fall motions. A simple and [effective heuristic approach](https://blog.ambianic.ai/2021/03/25/fall-detection.html) to detect a fall is to compare the angle between the spinal vectors of a person in before and after images from a fall sequence.

In many cases, this approach predicts true events. However, there are also scenarios where results are false positives. For example, when someone is leaning intentionally to tie shoes or squatting.

This approach also fails sometimes to predict true falls when the spinal vector angle between before and after frames does not meet a pre-configured threshold. For example when someone slides out of their bed/couch and is unable to hold their weight. They get stuck sitting on the floor unable to call for help. While this is not the most obvious example of a fall, it is the ground truth for seniors with specific medical conditions. 

In the process of testing the system with users in real-world settings, we discovered a number of challenges. Some of them are listed below:

### Distance from subject

While PoseNet and MoveNet were optimized for distances around 10-15 feet (3-4m), falls can occur in a bigger range from the camera lens. Because falls are unplanned events, we cannot expect people to position themselves in optimal distance right before they fall. Sometimes incidents happen very close to the camera, and sometimes further away.

### Camera angle

Pose detection models are trained mainly on data where the camera is positioned at eye level relative to the person(s) in the image or video. In home settings cameras are placed in a range of locations depending on the personal preference of the homeowner. The camera has to not only be functional, but it also needs to be “out of the way” and it has to “fit in” with the rest of the furniture in the room. Sometimes cameras are mounted near a ceiling corner and sometimes placed on a piece of furniture near the floor.

![image11](https://user-images.githubusercontent.com/2234901/131892933-5b64dcaf-71cb-4322-ab20-8900f6044beb.png)

_Example with a ceiling mounted camera._
 
![image9](https://user-images.githubusercontent.com/2234901/131892984-c0cb9382-6465-4be3-857a-673d1343161e.png)

_In this example with a camera angled down from a ceiling corner, PoseNet confused background objects with a person. MoveNet correctly focused on the person._

### Ambient lighting
Falls can happen any time. Day or night. For the current version of the system we assume minimal lighting exists in a home but we are well aware that we will have to drop these assumptions eventually as there are people who sometimes walk unintentionally(or intentionally) in their sleep through dark areas.

The base pose detection models perform well in cases with low lighting although they do have a limit. We actually saw examples where the ML models did well even when it was hard for a human eye to distinguish objects in a room with dim lighting.

![image1](https://user-images.githubusercontent.com/2234901/131893336-ed1f6d01-a5f6-49a4-8a0e-b854895f6b3b.png)

_Example of a low lighting room that makes it challenging for a human eye to distinguish key points, but the ML models did well._

![image2](https://user-images.githubusercontent.com/2234901/131893459-d928d26e-5a3a-47c6-9564-f2d8cc76278d.png)

_Another example with dim lighting. PoseNet got confused. MoveNet did well._

### Background objects
People personalize their spaces in a variety of ways. Some are minimalists in their choice of furniture and wall colors yet others are happier with a rainbow of colors and objects around them. The latter seems to present a notable challenge to computer vision models that have not seen such unique home decors.

![image8](https://user-images.githubusercontent.com/2234901/131893613-79c0ce55-d407-43e9-8288-8ed726a45556.png)

_In this example PoseNet got confused and placed pose key points on a vacuum cleaner. MoveNet did better._

![image7](https://user-images.githubusercontent.com/2234901/131893955-f8fbd23a-8c74-4999-a6fe-1cbc45356f69.png)

_In this example both models confused objects with person keypoints with. Although confidence scores were under 10%._


For the time being, we advise users to keep areas of high risk falls clear of clutter, but we are working on introducing a feedback loop that would allow users to help their local models learn about their personal space and reduce mistakes.

### Occlusions

It turns out that everyday items such as chairs and tables can be significant challenges to pose detection models. Both PoseNet and MoveNet suffer when occlusions block a certain part of the person whose fall we want to detect. Even in cases when the human eye can reasonably see and determine what position is a person in an image with occlusion, the ML models struggle. See examples below with PoseNet and MoveNet detections on a video frame sequence:

![Posenet](https://user-images.githubusercontent.com/2234901/132567974-f7ee8b91-34bf-4caf-a66f-53560baa9521.png)

_PoseNet suffers not only from confusing person with background objects, but it also loses track when occlusions block part of the person_

![Movenet](https://user-images.githubusercontent.com/2234901/132567938-1ebb0940-d7f1-41af-adf8-7ff6060d5771.png)

_MoveNet does not get easily confused by background objects, but it does have a problem with occlusions_


### Outdoor scenes
Doorsteps are a high-risk area for falls. We saw examples where ML models confused trees and pillars with people. The confidence scores for these detections were usually low (less than 10%) which alleviates the issue to some extent as we only recommend alerting when detections are with at least 50% confidence. However, this is another area where a user feedback loop would be appropriate to allow the local model to learn and avoid detecting incorrect objects in the particular home setting.

![image6](https://user-images.githubusercontent.com/2234901/131893832-602ad694-40b3-432a-ae1e-2c04e72b0dfa.png)

_Both PoseNet and MoveNet  confused a pillar of bricks with a person._

### Multiple people

Fall detection system alerts are mainly useful when people fall while there is no one around to help. However, there are situations when the people nearby are in a wheelchair or otherwise unable to assist. Therefore it is important to accurately distinguish between different individuals in a scene before analyzing their poses for a possible fall.

![image4](https://user-images.githubusercontent.com/2234901/131894202-220de1b5-0eb1-4dc2-a8a8-4c54b8bf782c.png)

_In an earlier frame (red lines) PoseNet incorrectly crossed key points between two people next to each other. In a different frame (green lines) it correctly focused on one person. MoveNet did not confuse the two people._

MoveNet’s approach to estimate a central body for the main subject and then assign relative weight of other key points as a function of distance from the central point helps with these situations. On our data it did better focusing on one main subject in an image. However the single pose version we tested did not address the issue of tracking the subject between frames when multiple people move around. It is an area that still requires work and testing of the upcoming multi-person tracking version of MoveNet.

### Other scenes
We continue to learn about new situations from our users who share their feedback with the [Open Source community](https://github.com/ambianic). As new challenging cases come in, we expand our model benchmark test suite and discuss problems that the ML system needs to learn to overcome. Along with this blog post, we also publish this [interactive notebook](https://github.com/ambianic/fall-detection/blob/main/MoveNet_Vs_PoseNet.ipynb) showing the latest results comparing PoseNet to MoveNet. 

## Future Work

The current system is helpful in many cases when unattended seniors fall and need assistance. That is an improvement over the status quo which requires a care team member to be always present in person in order to react adequately and in time. While wearable medical alert devices are a viable alternative and have been around for years, [research shows](https://www.theseniorlist.com/research/medical-alert-device-consumer-usage-study/) that seniors are not wearing one when needed.

We believe strongly that ambient intelligence has the potential to improve people's lives in a meaningful way. Although we are in the early days of proving this out, there is a growing community of users and supporters who continue to help us improve daily.

One immediate area of attention for the core Ambianic.ai team is to enable users to provide feedback through a beautiful UI app with minimal intuitive actions. That would enable local on-device transfer learning to train a [fall classification model](https://github.com/ambianic/fall-detection/issues/18) with improved accuracy, precision and recall metrics. 

The idea is to use the current simple heuristic model as a baseline for initial training of the fall classification DNN and gradually apply user feedback to further improve performance.

As users are presented with Fall Detections (from the current heuristic model) in the UI app, they can choose to Agree or Disagree with the detection. If they Agree, the detection sample will be labeled as positive and added to the local on-device training data set for the local fall classification model. If they Disagree, the sample will be labeled as negative classification and added to the local data set. Every 10th labeled sample will be added to a local test data set instead of the training dataset. 

Once there is a batch of 100 new samples collected on the device, TFLite Model Maker will run [on-device image classification transfer learning](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) on the fall classification model. When the fall classification model reaches 90% accuracy, it can replace the baseline heuristic model and continue to improve via user feedback as it comes.

Further down the road, we are planning to test out the promise of federated learning by enabling users to pool learning from multiple devices without sharing image data from cameras. 

If you want to do something about bringing positive change to your loved ones who need better care today or will need it tomorrow, then [try the Open Source Ambianic.ai product](https://ui.ambianic.ai/) and [join the project](https://github.com/orgs/ambianic/projects/1) now!

