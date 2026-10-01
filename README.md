# Face Emotion Recognition

A simple facial emotion recognition project using **OpenCV** for face detection
(Haar cascade) and **DeepFace** for emotion analysis. It supports both static
images and a live webcam feed.

> **Note:** I'm still exploring the field of image recognition and treating this
> as a learning project. I'm interested in training the model further and
> improving its accuracy over time, so expect this project to keep evolving.

## Features

- Detect faces in an image using a Haar cascade classifier
- Analyze the dominant emotion (happy, sad, angry, fear, surprise, neutral, disgust) with DeepFace
- Draw bounding boxes and the predicted emotion label on the frame
- Real-time emotion recognition from a webcam

## Requirements

- Python 3.x
- [opencv-python](https://pypi.org/project/opencv-python/)
- [deepface](https://pypi.org/project/deepface/)
- matplotlib

Install the dependencies:

```bash
pip install opencv-python deepface matplotlib
```

## Usage

Open `Code.ipynb` in Jupyter and run the cells.

- The image cells read a sample photo (e.g. `happy.jpg`), detect the face, and
  overlay the dominant emotion.
- The webcam cell opens your camera and labels emotions in real time. Press `q`
  to quit.

## Sample Images

The repo includes sample faces for each emotion: `angry.jpg`, `disgust.jpg`,
`fear.jpg`, `happy.jpg`, `neutral.jpg`, `sad.jpg`, `surprise.jpg`.

## Roadmap

As I keep learning, I plan to:

- Train and fine-tune the emotion model on datasets like FER2013 for better accuracy
- Experiment with different face detection methods beyond Haar cascades
- Improve real-time performance on the webcam feed

## References

- Haar cascade face detection (OpenCV)
- [DeepFace](https://github.com/serengil/deepface)
- [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)
