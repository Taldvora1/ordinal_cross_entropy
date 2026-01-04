import cv2
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
from glob import glob
from sklearn.model_selection import *
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
print("Necessary modules have been imported")


def split_diabetic_retinopathy_detection(set_name, have_p=False):
    # split to fixed train test
    labels_csv_path = f'data/{set_name}/train'
    df = pd.read_csv(labels_csv_path + '/trainLabels.csv')
    df['image'] = df['id_code']
    df['level'] = df['diagnosis']
    df['p'] = df['image']
    if have_p:
        df['p'] = df['image'].str.split('_').str[0]
    df_p_level = df.groupby(['p']).agg({'level': max}).reset_index()
    X_train, X_test, y_train, y_test = train_test_split(
        df_p_level, df_p_level['level'], stratify=df_p_level['level'], test_size=0.2
    )
    X_train['split'] = 'train'
    X_test['split'] = 'test'
    X = pd.concat([X_train, X_test])
    X = X.drop(columns=['level'])
    df = df.merge(X, how='left', on='p')
    df.to_csv(labels_csv_path + '/train_test_Labels.csv', index=False)


def restore_img(dir_list, outputPath, img_size):
    x_tr = []
    for img in dir_list:
        try:
            output_file = os.path.join(outputPath, os.path.splitext(img)[0] + '.png')
            img_arr = cv2.imread(output_file)[..., ::-1]
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            x_tr.append(resized_arr)
        except Exception as e:
            print(e)

    x_tr = np.asarray(x_tr)
    x_tr = x_tr / 255.0
    return x_tr


def restore_lbl(y_tr):
    y_tr = np.asarray(y_tr)
    y_tr = to_categorical(y_tr)
    return y_tr


def data_augmentation():
    """Create an ImageDataGenerator for data augmentation."""
    return ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        rotation_range=20,
        brightness_range=[0.8, 1.2]
    )


def get_data(set_name, opt='sample', save_new=True, top_r=False, fold_idx=None, return_val=False):
    """
    Load or preprocess image data for training, validation, and testing.
    - Uses the fold CSV as the single source of truth for the split.
    - Caches arrays to a **fold-aware** .npz file to avoid cross-fold contamination.

    :param set_name: dataset name
    :param opt: 'sample' for tiny demo, 'train' for full dataset
    :param save_new: if False and cache exists -> load cache; otherwise build & save
    :param top_r: kept for compatibility with load_data (not used here)
    :param fold_idx: None => default train/test CSV, int => use fold{idx}_train_val_Labels.csv
    :param return_val: if True, return (x_tr, y_tr, x_val, y_val, x_tst, y_tst), else (x_tr, y_tr, x_tst, y_tst)
    :return: training, validation (optional), and test arrays
    """

    # ---------- 1) Resolve paths ----------
    outputPath = f"data/{set_name}/{opt}"
    labels_csv_path = f"data/{set_name}/train"
    
    if fold_idx is not None:
        fold_path = f"{labels_csv_path}/fold{fold_idx}_train_val_Labels.csv"
    else:
        fold_path = f"{labels_csv_path}/train_test_Labels.csv"

    # Fold-aware cache path
    fold_tag = f"_fold{fold_idx}" if (fold_idx is not None) else "_default"
    cache_dir = os.path.join("data", set_name, f"{opt}_preproces")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"dataset{fold_tag}.npz")

    # If allowed and cache exists, load it and return
    if (not save_new) and os.path.exists(cache_path):
        return load_data(cache_path, top_r, return_val)

    # ---------- 2) Build from disk/CSV ----------
    if opt == 'sample':
        # Tiny demo: load all PNGs from the folder
        file_paths = glob(f"data/{set_name}/{opt}/*.png")
        dir_list = [os.path.basename(f) for f in file_paths]

        x_tr = restore_img(dir_list, outputPath, img_size=224)
        x_val = x_tr[:2]  # tiny validation set
        x_tst = x_tr[2:4]  # tiny test set

        y_tr = restore_lbl([0, 3, 1, 0, 2, 0, 3, 4])
        y_val = restore_lbl([1, 4])
        y_tst = restore_lbl([0, 2])

    else:
        # Read the split CSV
        df = pd.read_csv(fold_path)
        df['image'] = df['image'].astype(str).str.replace(r'\.png$', '', regex=True) + ".png"
        df = df.sort_index()

        # Extract file lists
        file_train = df[df['split'] == 'train']['image'].tolist()
        file_val = df[df['split'] == 'val']['image'].tolist()
        file_test = df[df['split'] == 'test']['image'].tolist()

        # Load images
        x_tr = restore_img(file_train, outputPath, img_size=224)
        x_val = restore_img(file_val, outputPath, img_size=224)
        x_tst = restore_img(file_test, outputPath, img_size=224)

        # Load labels
        y_tr = restore_lbl(df[df['split'] == 'train']['level'].to_numpy())
        y_val = restore_lbl(df[df['split'] == 'val']['level'].to_numpy())
        y_tst = restore_lbl(df[df['split'] == 'test']['level'].to_numpy())

        # ---------- 3) Sanity checks ----------
        assert len(file_train) == len(set(file_train)), "Duplicate filenames in TRAIN CSV!"
        assert len(file_val) == len(set(file_val)), "Duplicate filenames in VAL CSV!"
        assert len(file_test) == len(set(file_test)), "Duplicate filenames in TEST CSV!"
        assert x_tr.shape[0] == len(file_train), f"Train images {x_tr.shape[0]} != CSV rows {len(file_train)}"
        assert x_val.shape[0] == len(file_val), f"Val images {x_val.shape[0]} != CSV rows {len(file_val)}"
        assert x_tst.shape[0] == len(file_test), f"Test images {x_tst.shape[0]} != CSV rows {len(file_test)}"

    # ---------- 4) Save and return ----------
    save_data_with_val(x_tr, y_tr, x_val, y_val, x_tst, y_tst, cache_path)
    
    if return_val:
        return x_tr, y_tr, x_val, y_val, x_tst, y_tst
    else:
        return x_tr, y_tr, x_tst, y_tst


def save_data_with_val(x_tr, y_tr, x_val, y_val, x_tst, y_tst, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez(file_path, x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, x_tst=x_tst, y_tst=y_tst)
    print(f"Data saved to {file_path}")


def load_data(file_path, top_r, return_val=False):
    with np.load(file_path) as data:
        x_tr = data['x_tr']
        y_tr = data['y_tr']
        x_val = data.get('x_val', None)
        y_val = data.get('y_val', None)
        x_tst = data['x_tst']
        y_tst = data['y_tst']

    if top_r:
        y_tr_labels = np.argmax(y_tr, axis=1)
        df = pd.DataFrame({'class_label': y_tr_labels})
        sampled_indices = df.groupby('class_label').apply(
            lambda x: x.sample(min(len(x), 5000))
        ).index.get_level_values(1)
        x_tr = x_tr[sampled_indices]
        y_tr = y_tr[sampled_indices]

    print(f"Data loaded from {file_path}")
    
    if return_val and x_val is not None:
        return x_tr, y_tr, x_val, y_val, x_tst, y_tst
    else:
        return x_tr, y_tr, x_tst, y_tst

def save_data(x_tr, y_tr, x_tst, y_tst, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez(file_path, x_tr=x_tr, y_tr=y_tr, x_tst=x_tst, y_tst=y_tst)
    print(f"Data saved to {file_path}")


def load_data(file_path, top_r=False, return_val=False):
    with np.load(file_path) as data:
        x_tr = data['x_tr']
        y_tr = data['y_tr']
        x_val = data.get('x_val', None)
        y_val = data.get('y_val', None)
        x_tst = data['x_tst']
        y_tst = data['y_tst']

    if top_r:
        y_tr_labels = np.argmax(y_tr, axis=1)
        df = pd.DataFrame({'class_label': y_tr_labels})
        sampled_indices = df.groupby('class_label').apply(
            lambda x: x.sample(min(len(x), 5000))
        ).index.get_level_values(1)
        x_tr = x_tr[sampled_indices]
        y_tr = y_tr[sampled_indices]

    print(f"Data loaded from {file_path}")
    
    if return_val and x_val is not None:
        return x_tr, y_tr, x_val, y_val, x_tst, y_tst
    else:
        return x_tr, y_tr, x_tst, y_tst


def apply_smote(X, y, random_state=42):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def create_folds_csv(set_name, n_splits=5, have_p=False, test_frac=1/6, seed=42):
    """
    Generate multiple train/val CSV files for cross-validation.
    First splits data into holdout test (test_frac) and train_val (1-test_frac).
    Then applies StratifiedKFold on the train_val portion.
    
    :param set_name: dataset name
    :param n_splits: number of folds for CV
    :param have_p: whether to extract patient/group ID
    :param test_frac: fraction of data to reserve for final test set (default 1/6)
    :param seed: random seed for reproducibility
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    labels_csv_path = f'data/{set_name}/train'
    df = pd.read_csv(labels_csv_path + '/trainLabels.csv')
    df['image'] = df['id_code']
    df['level'] = df['diagnosis']
    df['p'] = df['image']
    if have_p:
        df['p'] = df['image'].str.split('_').str[0]

    # Group by patient and get max level per patient
    df_p_level = df.groupby(['p']).agg({'level': max}).reset_index()
    
    # Step 1: Split into train_val (5/6) and test (1/6) at patient level
    df_train_val, df_test = train_test_split(
        df_p_level,
        test_size=test_frac,
        stratify=df_p_level['level'],
        random_state=seed
    )
    
    print(f"Total patients: {len(df_p_level)}")
    print(f"Train+Val patients: {len(df_train_val)}")
    print(f"Holdout test patients: {len(df_test)}")
    
    # Step 2: Apply StratifiedKFold on train_val portion only
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['level'])):
        # Create fold split
        df_p_level_fold = df_p_level.copy()
        df_p_level_fold['split'] = ''
        
        # Mark train indices (from train_val subset)
        train_patients = df_train_val.iloc[train_idx]['p'].values
        df_p_level_fold.loc[df_p_level_fold['p'].isin(train_patients), 'split'] = 'train'
        
        # Mark validation indices (from train_val subset)
        val_patients = df_train_val.iloc[val_idx]['p'].values
        df_p_level_fold.loc[df_p_level_fold['p'].isin(val_patients), 'split'] = 'val'
        
        # Mark holdout test (same across all folds)
        test_patients = df_test['p'].values
        df_p_level_fold.loc[df_p_level_fold['p'].isin(test_patients), 'split'] = 'test'
        
        # Merge back to image level
        df_fold = df.merge(df_p_level_fold[['p', 'split']], how='left', on='p')
        
        # Save fold CSV
        fold_file = labels_csv_path + f'/fold{fold_idx}_train_val_Labels.csv'
        df_fold.to_csv(fold_file, index=False)
        
        # Print statistics
        train_count = (df_fold['split'] == 'train').sum()
        val_count = (df_fold['split'] == 'val').sum()
        test_count = (df_fold['split'] == 'test').sum()
        print(f"Fold {fold_idx}: Train={train_count}, Val={val_count}, Test={test_count} images")