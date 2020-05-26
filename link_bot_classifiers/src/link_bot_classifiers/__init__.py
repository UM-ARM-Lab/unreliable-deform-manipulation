from link_bot_classifiers import none_classifier, rnn_image_classifier


def get_model(model_class_name):
    if model_class_name == 'rnn':
        return rnn_image_classifier.RNNImageClassifier
    elif model_class_name == "none":
        return none_classifier.NoneClassifier
    else:
        raise ValueError
