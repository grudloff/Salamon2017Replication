# Salomon2017Replication
Replication of the Paper Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification by Salomon &amp; Bello. Implementation based on tensorflow.keras.

## repo outline

- setup : instructions for setting up the enviroment
- Functions:
  - evaluation.py : evaluation function
  - model.py : model building functions
  - preprocessing.py : data extraction, saving and loading from wav files with fixed 3 seconds length
  - preprocessing_augmented.py : data augmentation, extraction, saving and loading for wav+jams files
  - preprocessing_multi.py : data extraction, saving and loading from wav files
- Google colab notebook:
  - augmented_preprocessing.ipynb : preprocess dataset from jams
  - usk8:cnn_baseline.ipynb : preprocess dataset and train models on dataset with full lengths++
  - usk8:cnn_baseline_crop.ipynb : preprocess dataset and train models on dataset
- Desktop Notebooks:
  - usk8_cnn_salomon.ipynb : train model on augmented dataset (desktop)**
  - usk8_cnn_wavelet.ipynb : train improved model on augmented dataset (desktop)**


++ This trains the data as described on the paper. Random 3 seconds excerpts from each larger than 3 second sample for training, and average output over all possible excerpts on testing.

** Had to train this on a desktop as google colab couldn't handle the dataset on memory.
