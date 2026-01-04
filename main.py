import time
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from models_option import *
from preprocess import *
from sklearn.utils import class_weight
import warnings
from imblearn.over_sampling import SMOTE
import random

warnings.filterwarnings('ignore')
""" https://www.kaggle.com/c/diabetic-retinopathy-detection """

# Set seeds

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

set_name='APTOS 2019 Blindness Detection' # 'Diabetic Retinopathy Detection'

data_op='train'      #'sample'    'train'
loss_op_list=['oce_exponential']#'categorical_ce_beta_regularized','ordinal_ce_beta_regularized', 'cross_entropy_loss','ordinal_loss', 'weights_cross_entropy_loss']
 # 'inceptionv3', 'resnet152', 'resnet1', 'densenet121',    'densenet169',    'densenet201',    'mobilenet',    'alexnet', 'vgg19'
model_name_list = ['inceptionv3'] #'resnet152', 'resnet101', 'densenet121' #'inceptionv3' , 'resnet50' 'vgg16', 'vgg19' 'mobilenet' 'alexnet' 'densenet121' 'resnet101' 'resnet152'
callback_types = ['reduce_lr'] #'reduce_lr','early_stopping' '',
epochs = 25
n_splits = 5
smote_op_list=[False] # , True


# create folds once if missing 
labels_dir = f"data/{set_name}/train"
fold0_path = os.path.join(labels_dir, "fold0_train_val_Labels.csv")

if not os.path.exists(fold0_path):
    create_folds_csv(set_name=set_name, n_splits=5, have_p=True)
else:
    print("Folds already exist. Skipping creation.")



# Run combinations
for smote_op in smote_op_list:
    for fold_idx in range(n_splits):
        # Load data with validation set
        x_tr, y_tr, x_val, y_val, x_tst, y_tst = get_data(
            set_name, data_op, save_new=False, top_r=smote_op, 
            fold_idx=fold_idx, return_val=True
        )
        
        y_tr_classes = np.argmax(y_tr, axis=1)
        if smote_op:
            x_tr = x_tr.reshape(x_tr.shape[0], -1)
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            x_tr, y_tr_classes = smote.fit_resample(x_tr, y_tr_classes)
            x_tr = x_tr.reshape(-1, 224, 224, 3)

            y_tr = np.zeros((y_tr_classes.size, y_tr_classes.max() + 1))
            y_tr[np.arange(y_tr_classes.size), y_tr_classes] = 1

        for model_name in model_name_list:
            for loss_op in loss_op_list:
                model_eye = run_model(model_name, loss_op)
                for callback_type in callback_types:
                    print(f"Start model: {model_name}, loss: {loss_op}, callback: {callback_type}, smote: {smote_op}, fold: {fold_idx}")
                    callbacks_op = callbacks_define(callback_type)
                    
                    start_time = time.time()

                    # Train with validation data
                    hist = model_eye.fit(
                        x_tr, y_tr, 
                        validation_data=(x_val, y_val),  # Add validation data
                        epochs=epochs, 
                        callbacks=[callbacks_op]
                    )

                    last_epoch = len(hist.history['loss'])
                    
                    # Evaluate on train, validation, and test sets
                    df_metrics = evaluate_model(
                        model_eye, 
                        x_tr, y_tr, 
                        None, None,
                        data_op=f"fold{fold_idx}",
                        model_name=model_name,
                        loss_op=loss_op,
                        callback_type=callback_type,
                        last_epoch=last_epoch,
                        smote_op=smote_op,
                        set_name=set_name,
                        x_val=x_val,
                        y_val=y_val
                    )

                    end_time = time.time()
                    total_time_sec = end_time - start_time
                    total_time_min = total_time_sec / 60
                    df_metrics["train_time_min"] = total_time_min
                    plot_and_save_metrics(hist, f'fold{fold_idx}_{model_name}_{loss_op}_{callback_type}_{smote_op}', set_name)
                    print(f"Finish model: {model_name}, loss: {loss_op}, callback: {callback_type}, smote: {smote_op}, fold: {fold_idx}")

# Summarize results
df_final = pd.DataFrame()

for smote_op in smote_op_list:
    for model_name in model_name_list:
        for loss_op in loss_op_list:
            for callback_type in callback_types:
                for fold_idx in range(n_splits):
                    output_file = os.path.join(
                        os.getcwd(), "results", set_name,
                        f"fold{fold_idx}_{loss_op}_{callback_type}_{model_name}_{smote_op}_results.xlsx"
                    )
                    if os.path.exists(output_file):
                        df_result = pd.read_excel(output_file, sheet_name='Metrics')
                        df_result['fold'] = fold_idx  # Add fold identifier
                        df_final = pd.concat([df_final, df_result], ignore_index=True)
                    else:
                        print(f"Warning: File not found - {output_file}")

# Summarize results
df_final = pd.DataFrame()

for smote_op in smote_op_list:
    for model_name in model_name_list:
        for loss_op in loss_op_list:
            for callback_type in callback_types:
                for fold_idx in range(n_splits):
                    output_file = os.path.join(
                        os.getcwd(), "results", set_name,
                        f"fold{fold_idx}_{loss_op}_{callback_type}_{model_name}_{smote_op}_results.xlsx"
                    )
                    if os.path.exists(output_file):
                        df_result = pd.read_excel(output_file, sheet_name='Metrics')
                        df_result['fold'] = fold_idx  # Add fold identifier
                        df_final = pd.concat([df_final, df_result], ignore_index=True)
                    else:
                        print(f"Warning: File not found - {output_file}")

# Save the summary
df_final.to_csv(os.path.join(os.getcwd(), "results", set_name, f'{data_op}_summary_metrics.csv'), index=False)