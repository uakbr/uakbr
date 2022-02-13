Humanity has been waiting for self-driving cars for several decades. Thanks to the extremely fast evolution of technology, this idea recently went from “possible” to “commercially available in a Tesla”.

Deep learning is one of the main technologies that enabled self-driving. It’s a versatile tool that can solve almost any problem – it can be used in physics, for example, the proton-proton collision in the Large Hadron Collider, just as well as in Google Lens to classify pictures. Deep learning is a technology that can help solve almost any type of science or engineering problem. 

In this article, we’ll focus on deep learning algorithms in self-driving cars – convolutional neural networks (CNN). CNN is the primary algorithm that these systems use to recognize and classify different parts of the road, and to make appropriate decisions.  

Along the way, we’ll see how Tesla, Waymo, and Nvidia use CNN algorithms to make their cars driverless or autonomous. 

Contents
How do self-driving cars work?
– Deep learning in perception
– Deep learning in localization
– Deep learning for prediction
– Deep learning for decision-making
CNNs used for self-driving cars
– HydraNet
– ChauffeurNet
Reinforcement learning used for self-driving cars
– Reward function
– Q-learning
How do self-driving cars work?
The first self-driving car was invented in 1989, it was the Automatic Land Vehicle in Neural Network (ALVINN). It used neural networks to detect lines, segment the environment, navigate itself, and drive. It worked well, but it was limited by slow processing powers and insufficient data.

With today’s high-performance graphics cards, processors, and huge amounts of data, self-driving is more powerful than ever. If it becomes mainstream, it will reduce traffic congestion and increase road safety. 

Self-driving cars are autonomous decision-making systems. They can process streams of data from different sensors such as cameras, LiDAR, RADAR, GPS, or inertia sensors. This data is then modeled using deep learning algorithms, which then make decisions relevant to the environment the car is in. 

Self driving cars - pipeline
A modular perception-planning-action pipeline | Source
The image above shows a modular perception-planning-action pipeline used to make driving decisions. The key components of this method are the different sensors that fetch data from the environment. 

To understand the workings of self-driving cars, we need to examine the four main parts:

Perception 
Localization
Prediction
Decision Making
High-level path planning 
Behaviour Arbitration
Motion Controllers
Perception 
One of the most important properties that self-driving cars must have is perception, which helps the car see the world around itself, as well as recognize and classify the things that it sees. In order to make good decisions, the car needs to recognize objects instantly.

So, the car needs to see and classify traffic lights, pedestrians, road signs, walkways, parking spots, lanes, and much more. Not only that, it also needs to know the exact distance between itself and the objects around it. Perception is more than just seeing and classifying, it enables the system to evaluate the distance and decide to either slow down or brake. 

To achieve such a high level of perception, a self-driving car must have three sensors:

Camera
LiDAR
RADAR
Camera
The camera provides vision to the car, enabling multiple tasks like classification, segmentation, and localization. The cameras need to be high-resolution and represent the environment accurately.

In order to make sure that the car receives visual information from every side: front, back, left, and right, the cameras are stitched together to get a 360-degree view of the entire environment. These cameras provide a wide-range view as far as 200 meters as well as a short-range view for more focused perception. 

Self driving cars - camera
Self-driving car’s camera | Source
In some tasks like parking, the camera also provides a panoramic view for better decision-making. 

Even though the cameras do all the perception related tasks, it’s hardly of any use during extreme conditions like fog, heavy rain, and especially at night time. During extreme conditions, all that cameras capture is noise and discrepancies, which can be life-threatening. 

To overcome these limitations, we need sensors that can work without light and also measure distance.

LiDAR
LiDAR stands for Light Detection And Ranging, it’s a method to measure the distance of objects by firing a laser beam and then measuring how long it takes for it to be reflected by something.

A camera can only provide the car with images of what’s going around itself. When it’s combined with the LiDAR sensor, it gains depth in the images – it suddenly has a 3D perception of what’s going on around the car. 

So, LiDAR perceives spatial information. And when this data is fed into deep neural networks, the car can predict the actions of the objects or vehicles close to it. This sort of technology is very useful in a complex driving scenario, like a multi-exit intersection, where the car can analyze all other cars and make the appropriate, safest decision.

Self driving car - LiDAR
Object detection with LiDAR | Source
In 2019, Elon Musk openly stated that “anyone relying on LiDARs are doomed…”. Why? Well, LiDARs have limitations that can be catastrophic. For example, the LiDAR sensor uses lasers or light to measure the distance of the nearby object. It will work at night and in dark environments, but it can still fail when there’s noise from rain or fog. That’s why we also need a RADAR sensor.

RADARs
Radio detection and ranging (RADAR) is a key component in many military and consumer applications. It was first used by the military to detect objects. It calculates distance using radio wave signals. Today, it’s used in many vehicles and has become a primary component of the self-driving car. 

RADARs are highly effective because they use radio waves instead of lasers, so they work in any conditions. 

Self driving cars - lidar vs radar
Source
It’s important to understand that radars are noisy sensors. This means that even if the camera sees no obstacle, the radar will detect some obstacles. 

Self driving cars - lidar
Source
The image above shows the self-driving car (in green) using LiDAR to detect objects around and to calculate the distance and shape of the object. Compare the same scene, but captured with the RADAR sensor below, and you can see a lot of unnecessary noise.

Self driving cars - radar
Source
The RADAR data should be cleaned in order to make good decisions and predictions. We need to separate weak signals from strong ones; this is called thresholding. We also use Fast Fourier Transforms (FFT) to filter and interpret the signal. 

If you look at the below above, you’ll notice that the RADAR and LiDAR signals are point-based data. This data should be clustered so that it can be interpreted nicely. Clustering algorithms such as Euclidean Clustering or K means Clustering are used to achieve this task. 

Self driving cars - lidar and radar
Source
Localization
Localization algorithms in self-driving cars calculate the position and orientation of the vehicle as it navigates – a science known as Visual Odometry (VO).

VO works by matching key points in consecutive video frames. With each frame, the key points are used as the input to a mapping algorithm. The mapping algorithm, such as Simultaneous localization and mapping (SLAM), computes the position and orientation of each object nearby with respect to the previous frame and helps to classify roads, pedestrians, and other objects around. 

Self driving cars - localization
Source
Deep learning is generally used to improve the performance of VO, and to classify different objects. Neural networks, such as PoseNet and VLocNet++, are some of the frameworks that use point data to estimate the 3D position and orientation. These estimated 3D positions and orientations can be used to derive scene semantics, as seen in the image below. 

Self driving cars - localization
Source
Prediction
Understanding human drivers is a very complex task. It involves emotions rather than logic, and these are all fueled with reactions. It becomes very uncertain what the next action will be of the drivers or pedestrians nearby, so a system that can predict the actions of other road users can be very important for road safety. 

The car has a 360-degree view of its environment that enables it to perceive and capture all the information and process it. Once fed into the deep learning algorithm, it can come up with all the possible moves that other road users might make. It’s like a game where the player has a finite number of moves and tries to find the best move to defeat the opponent. 

The sensors in self-driving cars enable them to perform tasks like image classification, object detection, segmentation, and localization. With various forms of data representation, the car can make predictions of the object around it.

A deep learning algorithm can model such information (images and cloud data points from LiDARs and RADARs) during training. The same model, but during inference, can help the car to prepare for all the possible moves which involve braking, halting, slowing down, changing lanes, and so on. 

The role of deep learning is to interpret complex vision tasks, localize itself in the environment, enhance perception, and actuate kinematic maneuvers in self-driving cars. This ensures road safety and easy commute as well.

But the tricky part is to choose the correct action out of a finite number of actions. 

Decision-making
Decision-making is vital in self-driving cars. They need a system that’s dynamic and precise in an uncertain environment. It needs to take into account that not all sensor readings will be true, and that humans can make unpredictable choices while driving. These things can’t be measured directly. Even if we could measure them, we can’t predict them with good accuracy. 

Self driving cars - decision making
A self-driving car moving towards an intersection | Source
The image above shows a self-driving car moving towards an intersection. Another car, in blue, is also moving towards the intersection. In this scenario, the self-driving car has to predict whether the other car will go straight, left, or right. In each case, the car has to decide what maneuver it should perform to prevent a collision.

In order to make a decision, the car should have enough information so that it can select the necessary set of actions. We learned that the sensors help the car to collect information and deep learning algorithms can be used for localization and prediction. 

To recap, localization enables the car to know its initial position, and prediction creates an n number of possible actions or moves based on the environment. The question remains: which option is best out of the many predicted actions? 

When it comes to making decisions, we use deep reinforcement learning (DRL). More specifically, a decision-making algorithm called the Markov decision process (MDP) lies at the heart of DRL (we’ll learn more about MDP in a later section where we talk about reinforcement learning). 

Usually, an MDP is used to predict the future behavior of the road-users. We should keep in mind that the scenario can get very complex if the number of objects, especially moving ones, increases. This eventually increases the number of possible moves for the self-driving car itself. 

In order to tackle the problem of finding the best move for itself, the deep learning model is optimized with Bayesian optimization. There are also situations where the framework, consisting of both a hidden Markov model and Bayesian Optimization, is used for decision-making. 

In general, decision-making in self-driving cars is a hierarchical process. This process has four components:

Path or Route planning: Essentially, route planning is the first of four decisions that the car must make. Entering the environment, the car should plan the best possible route from its current position to the requested destination. The idea is to find an optimal solution among all the other solutions.  
Behaviour Arbitration: Once the route is planned, the car needs to navigate itself through the route. The car knows about the static elements, like roads, intersections, average road congestion and more, but it can’t know exactly what the other road users are going to be doing throughout the journey. This uncertainty in the behavior of other road users is solved by using probabilistic planning algorithms like MDPs.
Motion Planning: Once the behavior layer decides how to navigate through a certain route, the motion planning system orchestrates the motion of the car. The motion of the car must be feasible and comfortable for the passenger. Motion planning includes speed of the vehicle, lane-changing, and more, all of which should be relevant to the environment the car is in.  
Vehicle Control: Vehicle control is used to execute the reference path from the motion planning system. 
Self driving cars - decision making
Source
CNNs used for self-driving cars
Convolutional neural networks (CNN) are used to model spatial information, such as images. CNNs are very good at extracting features from images, and they’re often seen as universal non-linear function approximators. 

CNNs can capture different patterns as the depth of the network increases. For example, the layers at the beginning of the network will capture edges, while the deep layers will capture more complex features like the shape of the objects (leaves in trees, or tires on a vehicle). This is the reason why CNNs are the main algorithm in self-driving cars. 

The key component of the CNN is the convolutional layer itself. It has a convolutional kernel which is often called the filter matrix. The filter matrix is convolved with a local region of the input image which can be defined as:


Where: 

the operator * represents the convolution operation,
w is the filter matrix and b is the bias, 
x is the input,
y is the output. 
The dimension of the filter matrix in practice is usually 3X3 or 5X5. During the training process, the filter matrix will constantly update itself to get a reasonable weight. One of the properties of CNN is that the weights are shareable. The same weight parameters can be used to represent two different transformations in the network. The shared parameter saves a lot of processing space; they can produce more diverse feature representations learned by the network.

The output of the CNN is usually fed to a nonlinear activation function. The activation function enables the network to solve the linear inseparable problems, and these functions can represent high-dimensional manifolds in lower-dimensional manifolds. Commonly used activation functions are Sigmoid, Tanh, and ReLU, which are listed as follows:


It’s worth mentioning that the ReLU is the preferred activation function, because it converges faster compared to the other activation functions. In addition to that, the output of the convolution layer is modified by the max-pooling layer which keeps more information about the input image, like the background and texture. 

The three important CNN properties that make them versatile and a primary component of self-driving cars are:

local receptive fields, 
shared weights, 
spatial sampling. 
These properties reduce overfitting and store representations and features that are vital for image classification, segmentation, localization, and more.

Convolutional neural networks
Source
Next, we’ll discuss three CNN networks that are used by three companies pioneering self-driving cars:

HydraNet by Tesla
ChauffeurNet by Google Waymo
Nvidia Self driving car
HydraNet – semantic segmentation for self-driving cars 
HydraNet was introduced by Ravi et al. in 2018. It was developed for semantic segmentation, for improving computational efficiency during inference time.  

Self driving cars - semantic segmentation
Source
HydraNets is dynamic architecture so it can have different CNN networks, each assigned to different tasks. These blocks or networks are called branches. The idea of HydraNet is to get various inputs and feed them into a task-specific CNN network. 

Take the context of self-driving cars. One input dataset can be of static environments like trees and road-railing, another can be of the road and the lanes, another of traffic lights and road, and so on. These inputs are trained in different branches. During the inference time, the gate chooses which branches to run, and the combiner aggregates branch outputs and makes a final decision. 

In the case of Tesla, they have modified this network slightly because it’s difficult to segregate data for the individual tasks during inference. To overcome that problem, engineers at Tesla developed a shared backbone. The shared backbones are usually modified ResNet-50 blocks.

This HydraNet is trained on all the object’s data. There are task-specific heads that allow the model to predict task-specific outputs. The heads are based on semantic segmentation architecture like the U-Net. 

Self driving cars - hydranet
Source
The Tesla HydraNet can also project a birds-eye, meaning it can create a 3D view of the environment from any angle, giving the car much more dimensionality to navigate properly. It’s important to know that Tesla doesn’t use LiDAR sensors. It has only two sensors, a camera and a radar. Although LiDAR explicitly creates depth perception for the car, Tesla’s hydranet is so efficient that it can stitch all the visual information from the 8 cameras in it and create depth perception. 

Self driving cars - tesla hydranet
Source
ChauffeurNet: training self-driving car using imitation learning
ChauffeurNet is an RNN-based neural network used by Google Waymo, however, CNN is actually one of the core components here and it’s used to extract features from the perception system. 

The CNN in ChauffeurNet is described as a convolutional feature network, or FeatureNet, that extracts contextual feature representation shared by the other networks. These representations are then fed to a recurrent agent network (AgentRNN) that iteratively yields the prediction of successive points in the driving trajectory.

The idea behind this network is to train a self-driving car using imitation learning. In the paper released by Bansal et al “ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst”, they argue that training a self-driving car even with 30 million examples is not enough. In order to tackle that limitation, the authors trained the car in synthetic data. This synthetic data introduced deviations such as introducing perturbation to the trajectory path, adding obstacles, introducing unnatural scenes, etc. They found that such synthetic data was able to train the car much more efficiently than the normal data. 

Usually, self-driving has an end-to-end process as we saw earlier, where the perception system is part of a deep learning algorithm along with planning and controlling. In the case of ChauffeurNet, the perception system is not a part of the end-to-end process; instead, it’s a mid-level system where the network can have different variations of input from the perception system. 

Self driving cars - ChauffeurNet
Source
ChauffeurNet yields a driving trajectory by observing a mid-level representation of the scene from the sensors, using the input along with synthetic data to imitate an expert driver.  

Self driving cars - ChauffeurNet
Source
In the image above, the cyan path depicts the input route, green box is the self-driving car, blue dots are the agent’s past route or position, and green dots are the predicted future routes or positions. 

Essentially, a mid-level representation doesn’t directly use raw sensor data as input, factoring out the perception task, so we can combine real and simulated data for easier transfer learning. This way, the network can create a high-level bird’s eye view of the environment which ultimately yields better decisions. 

Nvidia self-driving car: a minimalist approach towards self-driving cars
Nvidia also uses a Convolution Neural Network as a primary algorithm for its self-driving car. But unlike Tesla, it uses 3 cameras, one on each side and one at the front.  See the image below. 

Convolutional neural networks NVIDIA
Source
The network is capable of operating inroads that don’t have lane markings, including parking lots. It can also learn features and representations that are necessary for detecting useful road features. 

Compared to the explicit decomposition of the problem such as lane marking detection, path planning, and control, this end-to-end system optimizes all processing steps at the same time. 

Better performance is the result of internal components self-optimizing to maximize overall system performance, instead of optimizing human-selected intermediate criteria like lane detection. Such criteria understandably are selected for ease of human interpretation, which doesn’t automatically guarantee maximum system performance. Smaller networks are possible because the system learns to solve the problem with a minimal number of processing steps.

Reinforcement learning used for self-driving cars
Reinforcement learning (RL) is a type of machine learning where an agent learns by exploring and interacting with the environment. In this case, the self-driving car is an agent. 

EXPLORE MORE APPLICATIONS OF RL
10 Real-Life Applications of Reinforcement Learning

We discussed earlier how the neural network predicts a number of actions from the perception data. But, choosing an appropriate action requires deep reinforcement learning (DRL). At the core of DRL, we have three important variables:

State describes the current situation in a given time. In this case, it would be a position on the road. 
Action describes all the possible moves that the car can make. 
Reward is feedback that the car receives whenever it takes a certain action. 
Generally, the agent is not told what to do or what actions to take. So far as we have seen, in supervised learning, the algorithm maps input to the output. In DRL, the algorithm learns by exploring the environment and each interaction yields a certain reward. The reward can be both positive and negative. The goal of the DRL is to maximize the cumulative rewards. 

In self-driving cars, the same procedure is followed: the network is trained on perception data, where it learns what decision it should make. Because the CNNs are very good at extracting features of representations from the input, DRL algorithms can be trained on those representations. Training a DRL algorithm on these representations can yield good results because these extracted representations are the transformation of higher-dimensional manifolds into simpler lower-dimensional manifolds. Training on lower representation yields efficiency which is required at the inference. 

One key point to remember is that self-driving cars can’t be trained in real-world scenarios or roads because they will be extremely dangerous. Instead, self-driving cars are trained on a simulator where there’s no risk at all. 

Some open-source simulators are:

CARLA
SUMMIT​​
AirSim
DeepDrive
Flow
Self driving car simulator - deepdrive
A snapshot from Voyage Deepdrive | Source
Voyage Deepdrive
A snapshot from Voyage Deepdrive | Source
These cars (agents) are trained for thousands of epochs with highly difficult simulations before they’re deployed in the real world. 

During training, the agent (the car) learns by taking a certain action in a certain state. Based on this state-action pair, it receives a reward. This process happens over and over again. Each time the agent updates its memory of rewards. This is called the policy. 

The policy is described as how the agent makes decisions. It’s a decision-making rule. The policy defines the behaviour of the agent at a given time. 

For every negative decision the agent makes, the policy is changed. So in order to avoid the negative rewards, the agent checks the quality of a certain action. This is measured by the state-value function. State-value can be measured using the Bellman Expectation Equation.

The Bellman expectation equation, along with Markov Decision Process (MDP), makes up the two core concepts of DRL. But when it comes to self-driving cars, we have to keep in mind that the observations from the perception data should be mapped with the appropriate action and not just map the underlying state to the action. This is where a partially observed decision process or a Partially Observable Markov Decision Process (POMDP) is required, which can make decisions based on the observation. 

Partially Observable Markov Decision Process used for self-driving cars
The Markov Decision Process gives us a way to sequentialize decision-making. When the agent interacts with the environment, it does so sequentially over time. Each time the agent interacts with the environment, it gives some representation of the environment state. Given the representation of the state, the agent selects the action to take, as in the image below. 


The action taken is transitioned into some new state and the agent is given a reward. This process of evaluating a state, taking action, changing states, and rewarding is repeated. Throughout the process, it’s the agent’s goal to maximize the total amount of rewards. 

Let’s get a more constructive idea of the whole process:

At a give time t, the state of the environment is at St
The agent observes the current state St and selects an action At
The environment is then transitioned into a new state St+1, simultaneously the agent is rewarded Rt
In a partially observable Markov decision process (POMDP), the agent senses the environment state with observations received from the perception data and takes a certain action followed by receiving a reward. 

The POMDP has six components and it can be denoted as POMDP M:= (I, S, A, R, P, γ), where, 

I: Observations 
S: Finite set of states
A: Finite set of actions
R: Reward function
P: transition probability function
γ – discounting factor for future rewards. 
The objective of DRL is to find the desired policy that maximizes the reward at each given time step or, in other words, to find an optimal value-action function (Q-function).  

Q-learning used for self-driving cars
Q-learning is one of the most commonly used DRL algorithms for self-driving cars. It comes under the category of model-free learning. In model-free learning, the agent will try to approximate the optimal state-action pair. The policy still determines which action-value pairs or Q-value are visited and updated (see the equation below). The goal is to find optimal policy by interacting with the environment while modifying the same when the agent makes an error. 

With enough samples or observation data, Q-learning will learn optimal state-action value pairs. In practice, Q-learning has been shown to converge to the optimum state-action values for a MDP with probability 1, provided that all actions in all states are infinitely available. 

Q-learning can be described in the following equation: 


where:

α ∈ [0,1] is the learning rate. It controls the degree to which Q values are updated at a given t.

Self driving cars - q learning
Source
It’s important to remember that the agent will discover the good and bad actions through trial and error.

Conclusion
Self-driving cars aim to revolutionize car travel by making it safe and efficient. In this article, we outlined some of the key components such as LiDAR, RADAR, cameras, and most importantly – the algorithms that make self-driving cars possible. 

While it’s promising, there’s still a lot of room for improvement. For example, current self-driving cars are at level-2 out of level-5 of advancement, which means that there still has to be a human ready to intervene if necessary. 

Few things need to be taken care of:

The algorithms used are not yet optimal enough to perceive roads and lanes because some roads lack markings and other signs.
The optimal sensing modality for localization, mapping, and perception still lack accuracy and efficiency.
Vehicle-to-vehicle communication is still a dream, but work is being done in this area as well.  
The field of human-machine interaction is not explored enough, with many open, unsolved problems.
Still, the technology we’ve developed so far is amazing. And with orchestrated efforts, we can ensure that self-driving systems will be safe, robust, and revolutionary.