# POS Tagger using PyTorch and UD_English-Atis Dataset

The purpose of this POS tagger is to categorize the parts-of-speech in English sentences by utilizing the UD_English-Atis dataset. To accomplish this task, a PyTorch implementation of an LSTM architecture is employed.

### Usage 
The POS tagger can be used for either training or prediction. To use it for prediction, simply run the script without any arguments:
`python3 model.py`

This will load a pre-trained model called pos_tagger.pth from the same location as the script and prompt you for input sentences. You can enter one sentence, and the model will output the predicted part-of-speech tags for each token.

I have used the train function to train this pos_tagger model, and have now commented it out. If you wish to train the model yourself, you can uncomment the train function and run the script. The model will be saved to the same location as the script.

# Results
The model achieved an overall accuracy of 89% on the test set of the UD_English-Atis dataset. The precision, recall, and F1-score for each of the datasets are shown in below.

Train Set
Accuracy: 0.8818751924537418
F1: 0.9206288209941081
Precision: 0.926253499128829
Recall: 0.936126201782184

Validation Set
Accuracy: 0.8811720194131272
F1: 0.9231126481922658
Precision: 0.9266586897111684
Recall: 0.9410848939683967

Test set
Accuracy: 0.8896858018542312
F1: 0.9266106581615378
Precision: 0.9314654970987201
Recall: 0.9414749413784709