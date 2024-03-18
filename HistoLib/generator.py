from .utils import get_dataframe, get_classes_labels, train_test_split

import random
import imageio
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical

def train_augmentations(percent_resize=0.25):
    """
    Train augmentations.
    The albumentations library is used to create multiple augmentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(p=0.2),
        A.RandomSizedCrop(min_max_height=(1000, 1200), height=1200, width=1600, p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                           val_shift_limit=10, p=.2),
        A.Resize(int(1200*percent_resize), int(percent_resize*1600)),
        A.ToFloat(max_value=255),
    ])

def test_augmentations(percent_resize=0.25):
    """
    Validation and Test augmentations. 
    Images are resized and normalized.
    """
    return A.Compose([
        A.Resize(int(1200*percent_resize), int(percent_resize*1600)),
        A.ToFloat(max_value=255),
])

class CustomDataGenerator(Sequence):
    def __init__(self, images, labels, num_classes, augmentations,
                 batch_size=8, 
                 shuffle_epoch=True):
        """
        Custom Generator. It loads the images from file and performs data augmentation.
        """
        
        self.num_classes = num_classes
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle_epoch = shuffle_epoch
        self.augment = augmentations
        
        random.seed(17)
        
    def __len__(self):
        # Number of batches in the generator
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        
        if (idx == 0) and (self.shuffle_epoch):            
            # Shuffle at first batch
            c = list(zip(self.images, self.labels))
            random.shuffle(c)
            self.images, self.labels = zip(*c)
            self.images, self.labels = np.array(self.images), np.array(self.labels)       
            
        # Get one batch
        bs = self.batch_size
        images = self.images[idx * bs : (idx+1) * bs]
        labels = self.labels[idx * bs : (idx+1) * bs]
        
        # Read images
        images = np.array([imageio.v3.imread(im) for im in images])                
        images = np.stack([self.augment(image=x)["image"] for x in images], axis=0)        
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels

    
    def show_generator(self, N=6):  
        used = set()
        fig, axs = plt.subplots(1,N, figsize=(20,4))
        for i in range(N):
            batch_idx = np.random.randint(0, len(self))
            batch = self[batch_idx]
            img_idx = np.random.randint(0, len(batch[0]))
            while (batch_idx, img_idx) in used:
                batch_idx = np.random.randint(0, len(self))
                batch = self[batch_idx]
                img_idx = np.random.randint(0, len(batch[0]))
            used.add((batch_idx, img_idx))
            img = batch[0][img_idx]
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(f'Class: {np.argmax(batch[1][img_idx])}')

            
            
def get_reproducible_ids(df, resolution):
    """
    Patient ids for reproducibility.
    
    Params
    ======
    :df: Dataset with all the image information ready to be splited in three sets: train / val / test.
    :resolution: One of '20x' or '40x'.
    
    Returns
    =======
    :df_train, df_val, df_test: Three dataframes split by patient_id.
    """
    
    if resolution == '20x':
        train_ids = [ 2,  3,  4,  5,  7,  8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 36, 37, 38, 39, 41, 42, 45]
        val_ids = [ 1,  6, 27, 32, 44]
        test_ids = [ 9, 13, 31, 40]
    else:
        train_ids = [ 2,  6,  8,  9, 10, 12, 13, 14, 16, 18, 19, 21, 22, 24, 28, 29, 31, 33, 34, 35, 36, 38, 40, 44]
        val_ids = [ 1,  4, 17, 26, 30, 37, 45]
        test_ids = [11, 15, 20, 25, 32, 43]
    
    df_train = df[df.patient_id.isin(train_ids)]
    df_val = df[df.patient_id.isin(val_ids)]
    df_test = df[df.patient_id.isin(test_ids)]
    
    return df_train, df_val, df_test
        

def get_patient_generators(resolution, 
                           batch_size = 8,
                           root_directory='data/images/',
                           dataset_csv = 'data/data.csv',
                           train_split = 0.8,
                           val_split = 0.1,
                           random_state = 17,
                           image_scale = 0.25,
                           reproducible = False,
                           debug = False
                          ):
    """
    It builds the data generators.
    
    Params
    ======
    :resolution: One of '20x' or '40x'.
    :batch_size: Default is 8.
    :root_directory: Directory containing the images.
    :dataset_csv: Path to csv with image information.
    :train_split: If reproducible is False, it sets the percentage of images to use as training.
    :val_split: If reproducible is False, it sets the percentage of images to use as validation. The rest of images will be used for testing the model.
    :image_scale: Images will be resized to this percentage during training.
    :reproducible: If true, it loads the patient ids used in the publication to create the train / val / test sets.
    :debug: If true, it outputs the number of images and different patients in each set.
    
    Returns
    =======
    :train_generator, val_generator, test_generator: Three custom generators that load the data.
    :class_names: An array with the name of the classes in the dataset.
    """

    # Get the dataframe of the filtered images
    df = get_dataframe(dataset_csv, resolution=resolution)
    
    # Get the labels associated with each image
    class_names, labels = get_classes_labels(root_directory, df['image_path'].values)
    df['targetclass'] = labels
    
    if reproducible:
        
        # Fixed ids so we can reproduce the splits
        df_train, df_val, df_test = get_reproducible_ids(df, resolution)
    else:
        
        # Create stratified splits without overlaping pacientes
        df_train, df_test = train_test_split(df, test_size = 1-train_split, random_state=random_state)
        df_test, df_val = train_test_split(df_test, test_size = round((val_split)/(1-train_split), 3), random_state=random_state)
    
    # Labels
    train_labels, val_labels, test_labels = df_train['targetclass'].values, df_val['targetclass'].values, df_test['targetclass'].values
    
    # Generators
    train_generator = CustomDataGenerator(df_train['image_path'].values, train_labels, augmentations=train_augmentations(percent_resize=image_scale), num_classes=len(class_names), batch_size=batch_size)
    val_generator = CustomDataGenerator(df_val['image_path'].values, val_labels, augmentations=test_augmentations(percent_resize=image_scale), num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    test_generator = CustomDataGenerator(df_test['image_path'].values, test_labels, augmentations=test_augmentations(percent_resize=image_scale), num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    
    ##### Debug
    if debug:
        imw, imh, _ = train_generator[0][0][0].shape
        print(f"{f'Images ({imw}x{imh})':<20}  Training: {len(train_labels):<3} | Validation: {len(val_labels):<3} | Test: {len(test_labels):<3} | Total: {len(labels):<3}")
        print(f"{'Patients':<20}  Training: {len(set(df_train['patient_id'])):<3} | Validation: {len(set(df_val['patient_id'])):<3} | Test: {len(set(df_test['patient_id'])):<3} | Total: {len(set(df['patient_id'])):<3}")

        for tclass in set(labels):
            cs = f'Class {class_names[tclass]:<6} (id {tclass})'
            tr, tv, te = len(train_labels[train_labels==tclass]), len(val_labels[val_labels==tclass]), len(test_labels[test_labels==tclass])
            print(f"{cs:<20}  Training: {tr:<3} | Validation: {tv:<3} | Test: {te:<3} | Total: {tr+tv+te:<3}")
    #####
    
    return train_generator, val_generator, test_generator, class_names