#creating two dir for training and testing
!mkdir test_labels train_labels

# lists the files inside 'annotations' in a random order (not really random, by their hash value instead)
# Moves the first 1000/9000 labels (20% of the labels) to the testing dir: `test_labels`
!ls data/Annotation/* | sort -R | head -1000 | xargs -I{} mv {} test_labels/


# Moves the rest of labels 8000 labels to the training dir: `train_labels`
!ls data/Annotation/* | xargs -I{} mv {} train_labels/