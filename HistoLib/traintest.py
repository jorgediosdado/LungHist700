import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def get_logdir(model_name, run_name='', base_log='logs'):
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = model_name+run_name
    
    return os.path.join(base_log, date, run_name)


def compile_model(model, num_classes, init_lr=1e-4):
    model.compile(optimizer=tf.keras.optimizers.Adam(init_lr), 
                  loss='categorical_crossentropy', 
                  metrics=[
                        tf.keras.metrics.CategoricalAccuracy(name=f'metrics/accuracy'),
                        tf.keras.metrics.TopKCategoricalAccuracy(2, name=f'metrics/top-2-accuracy'),
                        tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='metrics/F1-macro'),
                        tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes, name='metrics/AUC'),
                        tf.keras.metrics.Precision(name='metrics/precision'),
                        tf.keras.metrics.Recall(name='metrics/recall'),
                        tf.keras.metrics.PrecisionAtRecall(0.99, name='metrics/P@R_99'),
                        tf.keras.metrics.PrecisionAtRecall(0.95, name='metrics/P@R_95'),
                        tf.keras.metrics.PrecisionAtRecall(0.9, name='metrics/P@R_90'),
                        tfa.metrics.MatthewsCorrelationCoefficient(num_classes=num_classes, name='metrics/MCC')
                    ],
                 )
    return model


def train_model(model, train_generator, val_generator, class_weights, log_dir, 
                num_epochs = 2000, 
                patience = 25,
                patience_lr = 10):

    callbacks =[
           EarlyStopping(monitor='val_loss', restore_best_weights=False, patience=patience),
           ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, min_lr=1e-7),       
           ModelCheckpoint(log_dir, monitor=f"val_loss", save_best_only=True, save_weights_only=True),
           TqdmCallback(leave=False)
    ]
    
    history = model.fit(train_generator, epochs=num_epochs, verbose=0, callbacks=callbacks, validation_data=val_generator,class_weight=class_weights)
    
    print(f'Loading weights with best iteration...')
    model.load_weights(log_dir)
    
    return history


def plot_metrics(history, axs):
    min_loss = np.argmin(history.history['val_loss'])

    # Plotting training / validation loss
    axs[0].grid(alpha=0.5)
    axs[0].plot(history.history['loss'], label='Train')
    axs[0].plot(history.history['val_loss'], label='Validation')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].axvline(min_loss, alpha=0.5, c='k', linestyle='dashed', label='best epoch')
    axs[0].legend()

    # Plotting training / validation AUC
    axs[1].grid(alpha=0.5)
    axs[1].plot(history.history['metrics/AUC'], label='Train')
    axs[1].plot(history.history['val_metrics/AUC'], label='Validation')
    axs[1].set_title('ROC-AUC Evolution')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('ROC-AUC')
    axs[1].axvline(min_loss, alpha=0.5, c='k', linestyle='dashed', label='best epoch')
    axs[1].legend()
    
    
def test_model(model, test_generator, class_names, ax, conf_normalize=True):
    """
    It shows the confusion matrix and the test metrics.
    """

    test_predictions = model.predict(test_generator)
    ys = [t[1] for t in test_generator]
    test_labels = np.concatenate([np.argmax(t, 1) for t in ys])

    # Convert predictions to class labels
    predicted_labels = np.argmax(test_predictions, axis=1)
    
    # Calculate metrics
    auc = roc_auc_score(np.concatenate(ys), test_predictions, average='macro', multi_class='ovo')
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average='macro')
    recall = recall_score(test_labels, predicted_labels, average='macro')

    print(f"Test AUC: {auc:.2f}")
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test Recall: {recall:.2f}")

    # Ensure that labels are unique and match the confusion matrix
    labels = np.unique(np.concatenate((test_labels, predicted_labels)))

    # Visualize the confusion matrix
        
    ConfusionMatrixDisplay.from_predictions(test_labels, predicted_labels, display_labels=class_names, normalize='true' if conf_normalize else None, ax=ax)
    ax.set_title(f'Acc: {accuracy:4.2f}')
    
    
def metrics_and_test(history, model, test_generator, class_names, conf_normalize=True):
    """
    In a single figure, it shows the train / val curves and the test confusion matrix.
    """
    
    fig, axs = plt.subplots(1,3,figsize=(12,5))
    plot_metrics(history, axs[:2])
    test_model(model, test_generator, class_names, ax=axs[2], conf_normalize=conf_normalize)
    
    plt.tight_layout()
    plt.show()