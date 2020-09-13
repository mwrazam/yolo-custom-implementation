Object Detection and Classification for Reproduction as Scalable Vetor Graphics Using YOLO v3
Authors: Muhammad Azam and Zirui Li

## Steps to run code

1. Create and activate virtual environment to contain necessary libraries and dependencies
```
virtualenv env
source env/bin/activate
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Execute main script to run program
```
python main.py
```

## Project Definition
This project will look at the detection and classification of simple shape objects drawn by hand. The use case we are looking at is being able to take a photo of what is written onto a whiteboard or blackboard, and translate it to a scalable vector graphics format allowing the user to further edit in software such as Microsoft PowerPoint or Adobe Illustrator. This would allow the user to more easily digitize  notes, brainstorming work, or material presented in a classroom.

While the domain of objects that are useful in these types of scenarios is quite large, we will be focusing on detecting simple shapes such as circles, squares, hexagons, etc. For object detection and classification we are using the You Only Look Once (YOLO) algorithim presented in [X].  We will be developing our implementation of this algorithim, train and test it on the Google Quick, Draw! dataset, and then test its performance on real world scenarios.

The initial scope of the project will look at training our model to recognize the following shapes: circle, square, and hexagon.

## Dataset
The dataset that we will use to train and test our algorithim is the Google Quick, Draw! [X] dataset. It contains more than 50 million images across 375 different object classes. Quick, Draw! is presented as a game that is open and online for anyone to play. It presents users with text of the object it wants them to draw, and gives them 20 seconds to provide their drawing. In real-time, the neural network that Google has trainined on this dataset tries to predict what the user is drawing. After 6 objects, the game is over, and the success rate as well as the process the algorithim took in object classification is presented.

## References
[1] Fernandez-Fernandez, Raul, et al. "Quick, Stat!: A Statistical Analysis of the Quick, Draw! Dataset." arXiv preprint arXiv:1907.06417 (2019).
[2] Ha, David, and Douglas Eck. "A neural representation of sketch drawings." arXiv preprint arXiv:1704.03477 (2017).
