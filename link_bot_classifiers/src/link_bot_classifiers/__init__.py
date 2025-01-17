from link_bot_classifiers import none_classifier, nn_classifier, rnn_recovery_model


def get_model(model_class_name):
    if model_class_name == 'rnn':
        return nn_classifier.NNClassifier
    elif model_class_name == "none":
        return none_classifier.NoneClassifier
    elif model_class_name == "recovery":
        return rnn_recovery_model.RNNRecoveryModel
    else:
        raise ValueError
