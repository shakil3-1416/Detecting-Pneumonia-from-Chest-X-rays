# Detecting-Pneumonia-from-Chest-X-rays
Comprehensive Pipeline for Pneumonia Detection from Chest X-ray Images using CNNs: From Data Preprocessing to Model Evaluation


This code implements a complete pipeline for training and evaluating convolutional neural network (CNN) models for pneumonia detection from chest X-ray images. It begins by setting up the Kaggle API, downloading the dataset, and preparing the data by exploring its structure and visualizing image distributions across different datasets. Data augmentation techniques are applied to increase the diversity of the training dataset.

Next, a baseline CNN model is constructed and trained on the training dataset while being evaluated on the validation dataset. Hyperparameter tuning is performed using the Keras Tuner to optimize the model's architecture and training parameters.

Then, transfer learning techniques are employed using pre-trained CNN models such as VGG16, ResNet50, and InceptionV3. These models serve as feature extractors, with additional dense layers added on top for classification. The transfer learning models are trained and evaluated on the validation and test datasets.

Finally, the trained models are evaluated on the test dataset, and their accuracies are compared. Confusion matrices and classification reports are generated to assess model performance. Overall, the code showcases a comprehensive approach to building, training, and evaluating CNN models for pneumonia detection, covering data preprocessing, augmentation, model construction, hyperparameter tuning, and performance evaluation.
