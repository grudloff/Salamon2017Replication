import numpy as np
import os

from muda import load_jam_audio, replay
from librosa import resample

from preprocessing import extract_features, load_fold, assure_path_exists

augment_folders=["bgnoise", "drc", "pitch1", "pitch2", "stretch"]
original_folder="original"


def extract_fold(parent_dir, fold, augment_folders, bands=128, frames=128, channels=1, **kwargs):
    # extract features from original and augmented audio in one fold
    # This expects a folder for each augmention with a folder inside called jams containing the 
    # JAMS files. The agumentations are replicated through these files.

    features = np.empty(shape=[0, bands, frames, channels])  # shape : [samples, frames, bands]
    labels = np.empty(shape=0 ,dtype=int)
    
    for filename in os.listdir(os.path.join(parent_dir,fold)):
      if filename.endswith(".wav"):
        audio_path = os.path.join(parent_dir,fold,filename)
 
        filename = filename[:-4] # discard extension

        #extract original data
        jams_original_path = os.path.join(parent_dir, fold, original_folder, "jams", filename+".jams")
        jams_original = load_jam_audio(jams_original_path, audio_path)

        audio_original = jam_original.sandbox.muda._audio['y'] 
        orig_sr  = jam_original.sandbox.muda._audio['sr'] 
        audio_original = resample(audio_original, orig_sr, sr)
        features_yield = extract_features(audio_original, **kwargs)
        features = np.concatenate((features, features_yield))

        labels_yield = int(filename.split('-')[-3]) # filenames: [fsID]-[classID]-[occurrenceID]-[sliceID]
        labels = np.append(labels, labels_yield)

        #replay and extract data from augmentations
        for augment_folder in augment_folders:
          for i in range(4):
            if augment_folder is "pitch1":
              augmented_filename = filename + "_pitch" + str(i) 
            elif augment_folder is "pitch2":
              augmented_filename = filename + "_pitch3-" + str(i) 
            else:
              augmented_filename = filename + "_" + augment_folder + str(i) 

            jams_augmented_path = os.path.join(parent_dir, fold, augment_folder,"jams", augmented_filename+".jams")
            jams_augmented = load_jam_audio(jams_augmented_path, audio_path)
            jams_augmented = replay(jams_augmented, jams_original) # Apply augmentations

            audio_augmented = jams_augmented.sandbox.muda._audio['y'] 
            audio_augmented = resample(audio_augmented, orig_sr, sr)
            features_yield = extract_features(audio_augmented, **kwargs)
            features = np.concatenate((features, features_yield))

            labels = np.append(labels, labels_yield)

    return features, labels
    
    
def save_folds(data_dir, save_dir, **kwargs):
# use this to process the original and agmented audio files into numpy arrays
    assure_path_exists(save_dir)
    
    for k in range(1,10+1):
        fold_name = 'fold' + str(k)
        feature_file = os.path.join(save_dir, fold_name + '_x.npy')
        labels_file = os.path.join(save_dir, fold_name + '_y.npy')
        
        #Only save if file doesn't exist
        if not os.path.isfile(feature_file):

            print ("\nSaving " + fold_name)
            features, labels = extract_fold(data_dir, fold_name, augment_folders, **kwargs)
            
            print ("Features of", fold_name , " = ", features.shape)
            print ("Labels of", fold_name , " = ", labels.shape)
            

            np.save(feature_file, features, allow_pickle = True)
            print ("Saved " + feature_file)
            np.save(labels_file, labels, allow_pickle = True)
            print ("Saved " + labels_file)

            
def load_folds(load_dir, augmented_load_dir, validation_fold): 

    train_x = 0  # shape : [samples, frames, bands]
    train_y = np.empty(shape=0, dtype = int)

    # choose one fold from the remaining folds for training
    train_set = set(np.arange(9)+1)-set([validation_fold])
    test_fold = np.random.choice(list(train_set),1)[0] 
    train_set=train_set-set([test_fold])

    print("\n*** Train on", train_set, 
        "Validate on", validation_fold, "Test on", test_fold, "***")

    for k in range(1,10+1):
        fold_name = 'fold' + str(k)

        if k == validation_fold:
            val_x, val_y = load_fold(load_dir, fold_name)
        elif k == test_fold:
            test_x, test_y = load_fold(augmented_load_dir, fold_name) #TODO: should this be augmented or original?
        else:
            features, labels = load_fold(augmented_load_dir, fold_name)
            train_x = np.concatenate((train_x, features))
            train_y = np.append(train_y, labels)

    print("val shape: ", val_x.shape)
    print("test shape: ", test_x.shape)
    print("train shape: ", train_x.shape)

    return train_x, test_x, val_x, train_y, test_y, val_y
