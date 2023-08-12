import importlib
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            ECGNET_CNN_M1=dict(
                module_name="ecgnet_cnn_m1",
            ),
        )

    def get_model(self, class_names, model_name="ECGNET_CNN_M1", 
                  input_series=4096,input_channel=12,
                  weights_path=None):
        base_model_class = getattr(
            importlib.import_module(
                f"models.{self.models_[model_name]['module_name']}"
            ),
            model_name)

        model = base_model_class(input_series,input_channel,
                class_names=class_names)

        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        return model

