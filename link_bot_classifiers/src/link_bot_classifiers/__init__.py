from link_bot_classifiers import single_image_classifier, none_classifier


def get_model(model_class_name):
    if model_class_name == 'raster':
        return single_image_classifier.SingleImageClassifier
    elif model_class_name == "none":
        return none_classifier.NoneClassifier
    else:
        raise ValueError
