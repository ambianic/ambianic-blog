
For many adults, one of the most difficult decisions to make is how to care for an elderly parent or relative that needs assistance. The AARP has found that almost 90% of folks over the age of 65 prefer to remain independent by living out their golden years at home. 

Whether living alone or with family members, elderly parents need constant monitoring. Why? This is because as they age, their risk to potentially life-threatening accidents increases. 

In fact, a slew of researches reveal that seniors are more prone to fall than other age classes. Falls are the leading cause of fatal injury and the most common cause of nonfatal trauma-related hospital admissions among older adults.

In a recent [guest blog post for Linux Foundation AI & Data](https://lfaidata.foundation/blog/2021/01/14/people-fall-detection-via-privacy-preserving-ai/) we shared the background of the problem and current market solutions.

The Fall Detection algorithm fits well with the Ambianic framework of privacy preserving AI for home monitoring and automation. The following diagram illustrates the overall system architecture. 
End users install an Ambianic Box to constantly monitor a fall risk area of the house. If a fall is detected, family and caregivers are instantly notified that their loved one has fallen down.

![Fall Detection high level system architecture](https://user-images.githubusercontent.com/2234901/112542950-25d6d300-8d83-11eb-9048-feabd64de22d.png)

In the current design we use a combination of the [PoseNet 2.0](https://github.com/tensorflow/tfjs-models/tree/master/posenet) Deep Neural Network model and domain specific heuristics to estimate a fall occurance. The following diagram illustates the main steps.

[![Fall Detection AI flow](https://user-images.githubusercontent.com/2234901/112545190-ea89d380-8d85-11eb-8e2c-7a6b104d159e.png)](https://drive.google.com/file/d/1sr2OcEWsGzoxJb4PwCIXOuEo7a5ubAxG/view?usp=sharing)

As we work with families and caregivers to test the system in real world scenarious, we expect to develop better intuition for the key factors that determine a fall in a sequence of video frames. 
Eventually we expect to replace some of the current heuristics with learned models that are able to more precisely distinguish between true falls and non-falls (e.g. bending over or squating to tie shoes).

Ideas and constructive criticism are welcome. Feel free to join the discussion on [Slack](https://ambianicai.slack.com/join/shared_invite/zt-eosk4tv5-~GR3Sm7ccGbv1R7IEpk7OQ#/), open a [github issue](https://github.com/ambianic/fall-detection) or PR draft.

