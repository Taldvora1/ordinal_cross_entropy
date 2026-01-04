from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import Precision
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam
from loss_option import *
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

def callbacks_define(callback_type):
    if callback_type == 'early_stopping':
        return EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            verbose=1
        )

    elif callback_type == 'model_checkpoint':
        return ModelCheckpoint(
            'best_model.h5',
            monitor='_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )

    elif callback_type == 'reduce_lr':
        return ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=3,
            mode='max',
            verbose=1
        )

    else:
        raise ValueError(f"Unknown callback type: {callback_type}")

def choose_model(model_name):
    if model_name=='vgg19':
        from keras.applications.vgg19 import VGG19
        model_eye = VGG19(weights='imagenet', input_shape = (224, 224, 3), include_top = False)
    elif model_name == 'inceptionv3':
        from tensorflow.keras.applications import InceptionV3
        model_eye = InceptionV3(weights='imagenet', input_shape=(224, 224, 3), include_top=False, pooling='avg')
    elif model_name=='resnet50':
        from tensorflow.keras.applications import ResNet50
        model_eye = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=False, pooling='avg')
    elif model_name=='resnet152':
        from tensorflow.keras.applications import ResNet152
        model_eye = ResNet152(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    elif model_name == 'resnet1':
        from tensorflow.keras.applications import ResNet50
        from keras.layers import AveragePooling2D
        from tensorflow.keras.layers import Dense
        base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        # Add a new top layer
        x = base_model.output
        x = AveragePooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        # x = Dense(1024, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        predictions = Dense(5, activation='leaky_relu')(x)
        predictions = Dense(num_class, activation='softmax')(x)

        # This is the model we will train
        model_eye = Model(inputs=base_model.input, outputs=predictions)
    # DenseNet models
    elif model_name == 'densenet121':
        from tensorflow.keras.applications import DenseNet121
        model_eye = DenseNet121(weights='imagenet', input_shape=(224, 224, 3), include_top=False,pooling ='avg')
    elif model_name == 'densenet169':
        from tensorflow.keras.applications import DenseNet169
        model_eye = DenseNet169(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    elif model_name == 'densenet201':
        from tensorflow.keras.applications import DenseNet201
        model_eye = DenseNet201(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

    # MobileNet
    elif model_name == 'mobilenet':
        from tensorflow.keras.applications import MobileNet
        model_eye = MobileNet(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

    # AlexNet (Custom Implementation since it's not in Keras)
    elif model_name == 'alexnet':
        from keras.models import Sequential
        from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense
        model_eye = Sequential([
            Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(pool_size=(3, 3), strides=2),
            Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=2),
            Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=2),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(1000, activation='softmax')
        ])

    else:
        raise ValueError(f"Model name {model_name} is not recognized")

    return model_eye

def run_model(model_name,loss_op):
    model_eye = choose_model(model_name)
    for layers in model_eye.layers:
        layers.trainable = False
    x_eye = Flatten()(model_eye.output)
    prediction_eye = Dense(5, activation='leaky_relu')(x_eye)
    #prediction_eye = Dense(5, activation='softmax')(prediction_eye)

    model_eye = Model(inputs = model_eye.input, outputs = prediction_eye)
    if loss_op=='cross_entropy_loss':
        model_eye.compile(loss = cross_entropy_loss,
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    elif loss_op=='weights_cross_entropy_loss':
        model_eye.compile(loss = weights_cross_entropy_loss,
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    elif loss_op=='ordinal_loss':
        model_eye.compile(loss = ordinal_loss,
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    elif loss_op =='ordinal_cross_entropy_loss':
        model_eye.compile(loss = ordinal_cross_entropy_loss,
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    elif loss_op=='categorical_ce_beta_regularized':
        model_eye.compile(loss = categorical_ce_beta_regularized(num_classes=5, eta=0.85),
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    elif loss_op=='ordinal_ce_beta_regularized':
        model_eye.compile(loss = ordinal_ce_beta_regularized(num_classes=5, eta=0.5),
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    
    elif loss_op=='categorical_ce_poisson_regularized':
        model_eye.compile(loss = categorical_ce_poisson_regularized(num_classes=5, eta=0.85),
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
        
    elif loss_op=='categorical_ce_binomial_regularized':
        model_eye.compile(loss = categorical_ce_binomial_regularized(num_classes=5, eta=0.85),
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    
    elif loss_op=='categorical_ce_exponential_regularized':
        model_eye.compile(loss = categorical_ce_exponential_regularized(num_classes=5, eta=0.85),
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    
    elif loss_op=='oce_exponential':
        model_eye.compile(loss = oce_exponential(num_classes=5, eta=0.5, tau=1.0),
                         optimizer = 'adam',
                         metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
                             , run_eagerly=False) #run_eagerly=True is useful for debugging
    

    return model_eye


def evaluate_model(model, x_tr, y_tr, x_tst, y_tst, data_op, model_name, loss_op, 
                   callback_type, last_epoch, smote_op, set_name, x_val=None, y_val=None):
    """
    Evaluate the model and calculate metrics on training, validation, and testing datasets.

    :param model: The trained Keras model.
    :param x_tr: Training data features.
    :param y_tr: Training data labels.
    :param x_tst: Testing data features.
    :param y_tst: Testing data labels.
    :param data_op: String to prepend to each file name.
    :param model_name: Name of the model.
    :param loss_op: Loss function used.
    :param callback_type: Callback type used.
    :param last_epoch: Last training epoch.
    :param smote_op: Whether SMOTE was used.
    :param set_name: Dataset name.
    :param x_val: Validation data features (optional).
    :param y_val: Validation data labels (optional).
    :return: DataFrame with metrics for training, validation, and testing datasets.
    """
    output_dir = f'results/{set_name}'
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Predict on training data
    y_tr_pred = model.predict(x_tr)
    y_tr_pred_classes = np.argmax(y_tr_pred, axis=1)
    y_tr_true_classes = np.argmax(y_tr, axis=1)

    # Predict on validation data if provided
    if x_val is not None and y_val is not None:
        y_val_pred = model.predict(x_val)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        y_val_true_classes = np.argmax(y_val, axis=1)
    else:
        y_val_pred = None
        y_val_pred_classes = None
        y_val_true_classes = None

    # Predict on test data
    if x_tst is not None and y_tst is not None:
        y_tst_pred = model.predict(x_tst)
        y_tst_pred_classes = np.argmax(y_tst_pred, axis=1)
        y_tst_true_classes = np.argmax(y_tst, axis=1)
    else:
        y_tst_pred = None
        y_tst_pred_classes = None
        y_tst_true_classes = None

    # Calculate precision and recall for training
    precision_train = precision_score(y_tr_true_classes, y_tr_pred_classes, average=None, zero_division=0)
    recall_train = recall_score(y_tr_true_classes, y_tr_pred_classes, average=None, zero_division=0)

    # Calculate precision and recall for validation (if available)
    if y_val_pred_classes is not None:
        precision_val = precision_score(y_val_true_classes, y_val_pred_classes, average=None, zero_division=0)
        recall_val = recall_score(y_val_true_classes, y_val_pred_classes, average=None, zero_division=0)
    else:
        precision_val = None
        recall_val = None

    # Calculate precision and recall for test
    if y_tst_pred_classes is not None:
        precision_test = precision_score(y_tst_true_classes, y_tst_pred_classes, average=None, zero_division=0)
        recall_test = recall_score(y_tst_true_classes, y_tst_pred_classes, average=None, zero_division=0)
    else:
        precision_test = None
        recall_test = None

    # Create class labels
    class_labels = [f'Class {i}' for i in range(len(precision_train))]

    # Calculate metrics
    metrics = {
        'model_name': [model_name],
        'loss_op': [loss_op],
        'callback_type': [callback_type],
        'last_epoch': last_epoch,
        'smote': smote_op,
        'Train Accuracy': [accuracy_score(y_tr_true_classes, y_tr_pred_classes)],
        'Train AUC': [roc_auc_score(y_tr, y_tr_pred, multi_class='ovr')],
    }

    # Add validation metrics if available
    if y_val_pred_classes is not None:
        metrics['Val Accuracy'] = [accuracy_score(y_val_true_classes, y_val_pred_classes)]
        metrics['Val AUC'] = [roc_auc_score(y_val, y_val_pred, multi_class='ovr')]
    else:
        metrics['Val Accuracy'] = [None]
        metrics['Val AUC'] = [None]

    # Add test metrics if available
    if y_tst_pred_classes is not None:
        metrics['Test Accuracy'] = [accuracy_score(y_tst_true_classes, y_tst_pred_classes)]
        metrics['Test AUC'] = [roc_auc_score(y_tst, y_tst_pred, multi_class='ovr')]
    else:
        metrics['Test Accuracy'] = [None]
        metrics['Test AUC'] = [None]

    # Add loss metrics for test set
    if y_tst_pred is not None:
        metrics['cross_entropy_loss'] = [cross_entropy_loss(y_tst, y_tst_pred).numpy()]
        metrics['weights_cross_entropy_loss'] = [weights_cross_entropy_loss(y_tst, y_tst_pred).numpy()]
        metrics['ordinal_loss'] = [ordinal_loss(y_tst, y_tst_pred).numpy()]
        metrics['ordinal_cross_entropy_loss'] = [ordinal_cross_entropy_loss(y_tst, y_tst_pred).numpy()]

    # Add precision and recall to the metrics
    for i, label in enumerate(class_labels):
        metrics[f'Train Precision {label}'] = [precision_train[i]]
        metrics[f'Train Recall {label}'] = [recall_train[i]]
        
        if precision_val is not None:
            metrics[f'Val Precision {label}'] = [precision_val[i]]
            metrics[f'Val Recall {label}'] = [recall_val[i]]
        
        if precision_test is not None:
            metrics[f'Test Precision {label}'] = [precision_test[i]]
            metrics[f'Test Recall {label}'] = [recall_test[i]]

    # --- Cost Matrix Evaluation ---
    cost_matrix = np.array([
        [0, 4, 6, 8, 10],
        [4, 0, 4, 6, 8],
        [6, 4, 0, 4, 6],
        [8, 6, 4, 0, 4],
        [10, 8, 6, 4, 0]
    ])

    # Train cost
    train_costs = [cost_matrix[true, pred] for true, pred in zip(y_tr_true_classes, y_tr_pred_classes)]
    mean_train_cost = np.mean(train_costs)
    metrics['Train Cost'] = [mean_train_cost]

    # Validation cost (if available)
    if y_val_pred_classes is not None:
        val_costs = [cost_matrix[true, pred] for true, pred in zip(y_val_true_classes, y_val_pred_classes)]
        mean_val_cost = np.mean(val_costs)
        metrics['Val Cost'] = [mean_val_cost]
    else:
        metrics['Val Cost'] = [None]

    # Test cost (if available)
    if y_tst_pred_classes is not None:
        test_costs = [cost_matrix[true, pred] for true, pred in zip(y_tst_true_classes, y_tst_pred_classes)]
        mean_test_cost = np.mean(test_costs)
        metrics['Test Cost'] = [mean_test_cost]
    else:
        metrics['Test Cost'] = [None]

    # Create DataFrame for metrics
    df_metrics = pd.DataFrame(metrics)

    # Compute confusion matrices
    cm_tr = confusion_matrix(y_tr_true_classes, y_tr_pred_classes)
    
    df_cm_tr = pd.DataFrame(cm_tr,
                            columns=[f'Class {i}' for i in range(cm_tr.shape[1])],
                            index=[f'Class {i}' for i in range(cm_tr.shape[0])])

    output_file = f'{output_dir}/{data_op}_{loss_op}_{callback_type}_{model_name}_{smote_op}_results.xlsx'

    # Save metrics and confusion matrices to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
        df_cm_tr.to_excel(writer, sheet_name='Confusion_Matrix_Train')
        
        if y_val_pred_classes is not None:
            cm_val = confusion_matrix(y_val_true_classes, y_val_pred_classes)
            df_cm_val = pd.DataFrame(cm_val,
                                     columns=[f'Class {i}' for i in range(cm_val.shape[1])],
                                     index=[f'Class {i}' for i in range(cm_val.shape[0])])
            df_cm_val.to_excel(writer, sheet_name='Confusion_Matrix_Val')
        
        if y_tst_pred_classes is not None:
            cm_tst = confusion_matrix(y_tst_true_classes, y_tst_pred_classes)
            df_cm_tst = pd.DataFrame(cm_tst,
                                     columns=[f'Class {i}' for i in range(cm_tst.shape[1])],
                                     index=[f'Class {i}' for i in range(cm_tst.shape[0])])
            df_cm_tst.to_excel(writer, sheet_name='Confusion_Matrix_Test')

    return df_metrics

def plot_and_save_metrics(history, file_prefix, set_name):
    output_dir = f'graphs/{set_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    # Create a single figure with two subplots
    plt.figure(figsize=(14, 10))

    # Plot Loss
    plt.subplot(2, 1, 1)  # (rows, columns, panel number)
    plt.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss')
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot AUC
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history_dict['auc'], 'bo-', label='Training AUC')
    if 'val_auc' in history_dict:
        plt.plot(epochs, history_dict['val_auc'], 'ro-', label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.suptitle(file_prefix, fontsize=16)
    plt.tight_layout()

    # Save the figure with both subplots
    plt.savefig(f'{output_dir}/{file_prefix}_metrics.png')
    plt.close()