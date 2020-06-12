import glob
import os
import numpy as np

from librosa import power_to_db, load
from librosa.feature import melspectrogram

from pywt import dwt2

sr = 44100 # resample all to 44.1kHz
window_size = 1024 # ~23[mseg] at 44.1kHz


def extract_features(signal, normalize=False, wavelet=0):

    # handle less than 3 [seg]
    L = sr*3 # Total length for samples ~3[seg]
    signal_length = signal.shape[0]
    if signal_length < L:
        #pad by repeating signal
        signal = np.pad(signal, (0, L-signal_length), mode='wrap')
    elif signal_length > L:
        signal = signal[:L]

    # Calculate melspectrogram
    melspec = melspectrogram(signal, sr=sr, center=False,  #fmax = sr/2
                             hop_length=window_size, win_length=window_size, 
                             n_mels=128) # shape:[bands, frames]
                             
    # Transform to log scale and transpose
    melspec = power_to_db(melspec, ref=np.amax(melspec)).T # shape:[frames, bands]
    
    if normalize:
      melspec = (melspec - np.mean(melspec))/np.std(melspec)
    
    # 2D Discrete Wavelet Transform
    if wavelet != 0:
        LL, (LH, HL, HH) = dwt2(melspec, wavelet)
        melspec = np.stack([LL,LH,HL,HH],axis=-1) # shape: [frames, bands, 4]
    else:
        melspec = melspec[..., np.newaxis]
    
    # Reshape
    features = melspec[np.newaxis, ...] # shape : [1, frames, bands, channels]
    return features


def extract_fold(parent_dir, fold, frames=128, bands=128, channels=1, **kwargs):
# Extract features from one fold

    features = np.empty(shape=[0, bands, frames, channels])  # shape : [samples, frames, bands]
    labels = np.empty(shape=0 ,dtype=int)
        
    for filename in glob.glob(parent_dir+"/"+fold+"/*.wav"):

        # load signal
        signal = load(filename, sr=sr, duration=3)[0]

        #extract features
        features_yield = extract_features(signal, **kwargs)
        features = np.concatenate((features, features_yield))

        #extract label
        labels_yield = int(filename.split('-')[-3]) # filenames: [fsID]-[classID]-[occurrenceID]-[sliceID].wav 
        labels = np.append(labels, labels_yield)

    return features, labels


def save_folds(data_dir, save_dir, **kwargs):
# Preprocess all folds and save

    assure_path_exists(save_dir)
    for k in range(1,10+1):
        fold_name = 'fold' + str(k)
        print ("\nSaving " + fold_name)
        features, labels = extract_fold(data_dir, fold_name, **kwargs)

        print ("Features of", fold_name , " = ", features.shape)
        print ("Labels of", fold_name , " = ", labels.shape)

        feature_file = os.path.join(save_dir, fold_name + '_x.npy')
        labels_file = os.path.join(save_dir, fold_name + '_y.npy')
        np.save(feature_file, features, allow_pickle = True)
        print ("Saved " + feature_file)
        np.save(labels_file, labels, allow_pickle = True)
        print ("Saved " + labels_file)
        
    return


def load_folds(load_dir, validation_fold, bands=128, frames=128, channels=1): 
    #load all folds except the validation fold, and a random testing fold
    
    train_x = np.empty(shape=[0, bands, frames, channels])  # shape : [samples, frames, bands, channels]
    train_y = np.empty(shape=0, dtype=int)

    # take out validation from training set
    train_set = set(np.arange(1,10+1))-set([validation_fold])

    # take one random fold from the remaining for testing
    test_fold = np.random.choice(list(train_set),1)[0] 
    train_set = train_set-set([test_fold])

    print("\n*** Train on", train_set, 
          "Validate on", validation_fold, "Test on", test_fold, "***")

    for k in range(1,10+1):
        fold_name = 'fold' + str(k)
        feature_file = os.path.join(load_dir, fold_name + '_x.npy')
        labels_file = os.path.join(load_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file, allow_pickle=True)
        loaded_labels = np.load(labels_file, allow_pickle=True)

        if k == validation_fold:
            val_x,val_y = load_fold(load_dir, fold_name)
        elif k == test_fold:
            test_x, test_y = load_fold(load_dir, fold_name)
        else:
            features, labels = load_fold(load_dir, fold_name)
            train_x = np.concatenate((train_x, features))
            train_y = np.append(train_y, labels)

    print("val_x shape: ", val_x.shape)
    print("test_x shape: ", test_x.shape)
    print("train_x shape: ", train_x.shape)
    print("val_y shape: ", val_y.shape)
    print("test_y shape: ", test_y.shape)
    print("train_y shape: ", train_y.shape)

    return train_x, test_x, val_x, train_y, test_y, val_y


def load_fold(load_dir, fold_name):
    features_file = os.path.join(load_dir, fold_name + "_x.npy)
    labels_file = os.path.join(load_dir, fold_name + "_y.npy)  
    features = np.load(features_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    return features, labels
                               
                               
def assure_path_exists(path):
    # checks if path exists, if it dosen't it is created
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    return
