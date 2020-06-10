from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np

num_labels = 10

def evaluate(model, val_x, val_y):
    # Function for evaluating model through ROC, Accuracy ,F1 score and confusion matrix 
    
    y_prob = model.predict(val_x, verbose=0)
    y_pred = np.argmax(y_prob, 1)
    y_true = val_y


    roc = roc_auc_score(y_true, y_prob, multi_class = 'ovr', labels = range(num_labels))
    print("ROC:",  round(roc,3))

    # evaluate the model
    score, accuracy = model.evaluate(val_x, batch_size=100, verbose=0)
    print("\nAccuracy = {:.2f}".format(accuracy))

    # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print("F-Score:", round(f,2))
    
    cm = confusion_matrix(y_true, y_pred)
    
    return roc, accuracy, cm
