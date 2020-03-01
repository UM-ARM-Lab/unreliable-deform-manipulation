from link_bot_classifiers import raster_classifier, none_classifier


def get_model(model_class_name):
    if model_class_name == 'raster':
        return raster_classifier.RasterClassifier
    elif model_class_name == "none":
        return none_classifier.NoneClassifier
    else:
        raise ValueError
