from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2

def resnet_model(num_classes, input_shape):
    inputs = layers.Input(input_shape)
    
    base_model = ResNet50V2(include_top = False, weights='imagenet', pooling='avg', input_tensor=inputs)

    # Add a dense layer
    x = layers.Dense(256, activation='relu', name='prev_dense')(base_model.output)
    
    # Add another dense layer
    x = layers.Dropout(0.5, name='dpout')(x)
    salidas = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)
    model1 = Model(inputs = inputs, outputs = salidas)
    
    return model1, 'ResNet50V2'
        

def get_model(generator, model_name='ResNet50'):
    assert model_name in ['ResNet50']
    
    num_classes = generator.num_classes
    input_shape = generator[0][0][0].shape
    
    if model_name == 'ResNet50':
        return resnet_model(num_classes, input_shape)