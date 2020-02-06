from link_bot_classifiers import raster_classifier, none_classifier, feature_classifier


def get_model_module(model_class_name):
    if model_class_name == 'raster':
        return raster_classifier
    elif model_class_name == "none":
        return none_classifier
    elif model_class_name == "feature":
        return feature_classifier
