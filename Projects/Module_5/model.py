import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.preprocessing import image_dataset_from_directory

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # Confirm model is using GPU

### Training data
training_dir = r'C:\Users\ajvas\Documents\GitHub\CSE450\Projects\Module_5\content\training'
image_size = (100, 100)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2
        )
validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2
        )

train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size = image_size,
        subset="training",
        batch_size=32,
        class_mode='sparse',
        seed=42,shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
        training_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='sparse',
        subset="validation",
        seed=42,
        shuffle=False)

target_names = ['Speed_20', 'Speed_30', 'Speed_50', 'Speed_60', 'Speed_70',
               'Speed_80','Speed_Limit_Ends', 'Speed_100', 'Speed_120', 'Overtaking_Prohibited',
               'Overtakeing_Prohibited_Trucks', 'Priority', 'Priority_Road_Ahead', 'Yield', 'STOP',
               'Entry_Forbidden', 'Trucks_Forbidden', 'No_Entry(one-way traffic)', 'General Danger(!)', 'Left_Curve_Ahead',
               'Right_Curve_Ahead', 'Double_Curve', 'Poor_Surface_Ahead', 'Slippery_Surface_Ahead', 'Road_Narrows_On_Right',
               'Roadwork_Ahead', 'Traffic_Light_Ahead', 'Warning_Pedestrians', 'Warning_Children', 'Warning_Bikes',
               'Ice_Snow', 'Deer_Crossing', 'End_Previous_Limitation', 'Turning_Right_Compulsory', 'Turning_Left_Compulsory',
               'Ahead_Only', 'Straight_Or_Right_Mandatory', 'Straight_Or_Left_Mandatory', 'Passing_Right_Compulsory', 'Passing_Left_Compulsory',
               'Roundabout', 'End_Overtaking_Prohibition', 'End_Overtaking_Prohibition_Trucks']


### Model Prep
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(43, activation='softmax'))
model.add(layers.Dropout(0.2))
model.summary()


### Model
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(
    train_generator, 
    epochs=5, 
    validation_data=validation_generator
)


### Show model statistics
plt.figure(figsize=(10, 10))
images, labels = next(train_generator)
batch_size = images.shape[0]

for i in range(min(9, batch_size)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow((images[i] * 255).astype("uint8"))
    plt.title(int(labels[i]))
    plt.axis("off")
plt.show()

plt.show()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}")

predictions = model.predict(validation_generator)
predictions_new = np.argmax(predictions,axis=1)

# Confusion matrix
cm = confusion_matrix(validation_generator.classes, predictions_new)
print(f"Confusion Matrix: \n{cm}")
cmd = ConfusionMatrixDisplay(cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(13,13))
cmd.plot(ax=ax, cmap="Blues", values_format='.5g')

print(f"Classification Report: \n{classification_report(validation_generator.classes, predictions_new, target_names=target_names)}") 


### mini holdout test
test_dir = r'C:\Users\ajvas\Documents\GitHub\CSE450\Projects\Module_5\content\mini_holdout'

filenames = sorted([
    f for f in os.listdir(test_dir)
    if os.path.isfile(os.path.join(test_dir, f)) and f.endswith('.jpg')
])

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels=None,
    shuffle=False,
    image_size=image_size,
    batch_size=32
)
probabilities = model.predict(test_dataset)
predictions = [np.argmax(probas) for probas in probabilities]

# Test against answer
answers_url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/roadsigns/mini_holdout_answers.csv'
answers_df = pd.read_csv(answers_url)

preds_df = pd.DataFrame({
    'Filename': filenames,
    'Predicted': predictions
})

merged_df = pd.merge(answers_df, preds_df, on='Filename')
y_true = merged_df['ClassId']
y_pred = merged_df['Predicted']

print("\nHoldout Accuracy:", accuracy_score(y_true, y_pred))
print("\nHoldout Classification Report:\n", classification_report(y_true, y_pred))


### holdout csv
test_dir = r'C:\Users\ajvas\Documents\GitHub\CSE450\Projects\Module_5\content\holdout'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        classes=['holdout'],
        target_size=image_size,
        class_mode='sparse',
        shuffle=False
)

probabilities = model.predict(test_generator)
predictions = [np.argmax(probas) for probas in probabilities]

my_predictions = pd.DataFrame({'Predicted': predictions})
my_predictions.to_csv(path_or_buf="Projects/Module_5/team5-module5-predictions.csv", index=False)
