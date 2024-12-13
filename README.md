# forecast_cat_dog_project
use  Convolutional Neural Network to forecast_cat_dog_project

## 1. Installations

This project was implemented in Python using TensorFlow and Keras libraries. Below are the required libraries:

- tensorflow
- numpy
- keras

To install these libraries, run the following:
```bash
pip install tensorflow numpy
```

## 2. Project Motivation

The goal of this project is to create and train a Convolutional Neural Network (CNN) capable of classifying images as either a cat or a dog. The dataset consists of two classes: cats and dogs, with 8000 training images and 2000 test images.

This project aims to:
1. Preprocess image data to standardize inputs for the CNN.
2. Build and train a CNN with appropriate layers for feature extraction and classification.
3. Evaluate the model's performance on test data.
4. Demonstrate single image prediction functionality.

## 3. File Descriptions

This repository contains the following key files:

- **`CNN for Image Classification.ipynb`**: Python file containing the CNN implementation and training logic.
- **`dataset/training_set`**: Directory containing the training images categorized into two folders: `cats` and `dogs`.
- **`dataset/test_set`**: Directory containing the test images categorized into two folders: `cats` and `dogs`.
- **`dataset/single_prediction`**: Directory with sample images for single prediction testing.

## 4. Steps

### Part 1: Data Preprocessing
#### Training Set:
- Images are rescaled to a range of 0 to 1.
- Augmentation techniques like shear, zoom, and horizontal flipping are applied.

#### Test Set:
- Images are only rescaled to a range of 0 to 1.

### Part 2: Building the CNN
1. **Convolution**:
   - The CNN starts with a convolutional layer using 32 filters of size 3x3 and ReLU activation.

2. **Pooling**:
   - Max pooling with a pool size of 2x2 is applied to reduce dimensionality.

3. **Additional Layers**:
   - A second convolutional and pooling layer is added to improve feature extraction.

4. **Flattening**:
   - The output is flattened into a single vector for the fully connected layer.

5. **Full Connection**:
   - A dense layer with 128 units and ReLU activation is used.
   - Dropout is applied to prevent overfitting.

6. **Output Layer**:
   - A dense layer with 1 unit and sigmoid activation is used for binary classification.

### Part 3: Training the CNN
- **Compilation**:
  - The CNN is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.
- **Training**:
  - The model is trained for 25 epochs on the training set and validated on the test set.

### Part 4: Making Single Predictions
- A function is implemented to load a single image, preprocess it, and predict whether the image contains a cat or a dog.

## 5. Results

### Training Results:
- The model achieved a training accuracy of approximately 86.26% and a validation accuracy of 80.10% after 25 epochs.

### Single Prediction Results:
- The model successfully classified example images:
  - `dataset/single_prediction/cat_or_dog_1.jpg`: Dog
  - `dataset/single_prediction/cat_or_dog_2.jpg`: Cat

## 6. Licensing, Authors, Acknowledgements, etc.

- The dataset used for this project is sourced from the Kaggle "Dogs vs. Cats" dataset.
- This project is for educational purposes.
- Feel free to use or modify the code with proper attribution.

## 7. How to Run the Project
1. Place your dataset in the appropriate directories (`training_set`, `test_set`, and `single_prediction`).
2. Run the script `cnn_model.py` to train and evaluate the model.
3. Use the `make_prediction` function for single image predictions by providing the image path.

For any questions or issues, please open a GitHub issue or reach out.
