# Gesture Control Minecraft

This project aims to enable intuitive control of Minecraft through hand gestures, using real-time hand tracking and a custom-trained cv model.

## Project Background and Motivation

Through this project I was hoping to explore real time systems to control time sensitive and important tasks. The system built here can be used (directly!) for other projects to control real world objects like vehicles, bots, etc. This is more of a foundational layer really.

Do note that I have only used 100 data points for each hand gesture so the more data the better!

## Technical Overview

### Gesture Detection

Hand landmark detection is performed using **MediaPipe Hand Landmarker**. This library provides 21 key points for each hand, with coordinates already normalized, which simplified the data processing pipeline significantly.

### Data Collection

A custom script (`cv_mp_detection.py`) is used to capture hand landmark data in real-time from a webcam. This data is then formatted and saved into a CSV file (`data/hand_gestures.csv`). Each row in the dataset includes:
-   `sample_id`: Unique identifier for each recorded sample.
-   `label`: The name of the gesture being performed.
-   `x1, y1, ..., x21, y21`: The normalized (x, y) coordinates for each of the 21 hand landmarks.
_basically 42 landmarks to recognize and train off of_

### Model Training

The core of the gesture recognition system is a neural network implemented in **PyTorch**.

**Initial Test (NumPy):**
An initial attempt was made to construct a basic neural network using raw NumPy for a more deeper understanding of the underlying mechanics. This involved:
-   Two linear layers.
-   A ReLU activation function.
-   Custom implementations for `softmax` and `CrossEntropyLoss`.
-   A rudimentary backward pass.

While insightful, this approach highlighted the benefits of using a dedicated deep learning framework for efficiency and stability.

**PyTorch Implementation:**
The project transitioned to a more streamlined and scalable PyTorch model (`train_and_test.py`), featuring:
-   A simple architecture: `nn.Linear(42, 128)` -> `nn.ReLU()` -> `nn.Linear(128, 3)`.
-   Integration with `pandas` for data loading and preprocessing.
-   `sklearn.model_selection.train_test_split` for dataset partitioning.
-   `torch.utils.data.TensorDataset` and `DataLoader` for efficient batch processing during training.
-   `nn.CrossEntropyLoss` as the loss function.
-   `torch.optim.Adam` as the optimizer.

The model is trained to classify detected hand gestures based on the collected landmark data. (duh)

It was fun to build (and pretty easy!).

### Minecraft Integration

Once the gesture recognition model is trained, `pynput` is utilized to translate detected gestures into in-game commands or actions within Minecraft, providing a hands-free control experience. The goal is to ensure a smooth control experience, aiming for responsiveness even when Minecraft is limited to 30 FPS.

## Current Features

The project currently supports the recognition of the following hand gestures:
-   **Palm Open**
-   **Closed Fist**
-   **Point Up**

These gestures have been successfully integrated to control Minecraft.

## Setup and Installation

To get started with this project, ensure you have Python installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/Gesture_Control_Minecraft.git
    cd Gesture_Control_Minecraft
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.\.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` should contain `opencv-python`, `mediapipe`, `pandas`, `torch`, `scikit-learn`, and `pynput`.)*

4.  **Download the MediaPipe Hand Landmarker model:**
    Ensure `hand_landmarker.task` is in the project root directory. You can download it from [MediaPipe's official site](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

## Usage

1.  **Data Collection:** Run `cv_mp_detection.py` to collect your own hand gesture data. Follow the prompts in the console.
2.  **Model Training:** Execute `train_and_test.py` to train the PyTorch model on your collected data.
3.  **Inference/Game Control:** Run `inference.py` (or a similar script that integrates the trained model and `pynput`) to control Minecraft with your gestures.

## Future Enhancements

-   Expanding the dataset with a wider variety of gestures and more samples.
-   Exploring more complex neural network architectures for improved accuracy. (though we got 98.33% haha!!!!)
-   Implementing real-time inference in a lower-level language like C++ for performance gains.
-   Adding more sophisticated game control mappings and customization options.
