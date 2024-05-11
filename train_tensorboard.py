
from src.utils import prepare_batch_dataset
from src.utils import callbacks
from src import config
from src.network import MobileNet
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import pickle
from tensorflow.keras.callbacks import TensorBoard

# build the training and validation dataset pipeline
print("[INFO] building the training and validation dataset...")
train_ds = prepare_batch_dataset(
	config.TRAIN_DATA_PATH, config.IMAGE_SIZE, config.BATCH_SIZE
)
val_ds = prepare_batch_dataset(
	config.VALID_DATA_PATH, config.IMAGE_SIZE, config.BATCH_SIZE
)
# train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# build the output path if not already exists
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)
     
# initialize the callbacks, model, optimizer and loss
print("[INFO] compiling model...")
callbacks = callbacks()
model = MobileNet.build(
	width=config.IMAGE_SIZE,
	height=config.IMAGE_SIZE,
	depth=config.CHANNELS,
	classes=config.N_CLASSES
)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LR_INIT)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# compile the model
model.compile(
	optimizer=optimizer,
	loss=loss,
	metrics=["accuracy"]
)

# evaluate the model initially
(initial_loss, initial_accuracy) = model.evaluate(val_ds)
print("initial loss: {:.2f}".format(initial_loss))
print("initial accuracy: {:.2f}".format(initial_accuracy))
# train the image classification network
print("[INFO] training network...")

# Create TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=config.TENSORBOARD_LOG_DIR,
    histogram_freq=1,  # Log histogram visualizations every 1 epoch
    write_graph=True,
    write_images=True,
)



# to saved the models
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
	os.path.join(config.EACH_EPOCH, "model_checkpoint_{epoch:02d}.h5"),
	monitor = 'val_loss',
	save_best_only = False,
	save_weights_only = False,
	mode = 'auto',
	save_freq = 'epoch'

)

# resume trainingng from the last epoch
print(config.EACH_EPOCH)
if os.listdir(config.EACH_EPOCH):
    #latest_model = max(os.listdir(config.EACH_EPOCH)) #till epoch 99th
    latest_model = max(os.listdir(config.EACH_EPOCH), key=lambda x: int(x.split("_")[-1].split(".")[0])) # from epoch 100th.
    print("latest_model", latest_model)
    initial_epoch = int(latest_model.split("_")[-1].split(".")[0])
    print("initial epoch", initial_epoch)
    print(f"Resuming training from epoch {initial_epoch}...")
    model.load_weights(os.path.join(config.EACH_EPOCH, latest_model))
else:
    initial_epoch = 0
    
all_history = []

for epoch in range(initial_epoch, config.NUM_EPOCHS):
	history = model.fit(
		train_ds,
		epochs=config.NUM_EPOCHS,
	    initial_epoch=initial_epoch,
		validation_data=val_ds,
		callbacks= [callbacks, tensorboard_callback, model_checkpoint]
	)

   # Append the history object to the list
	all_history.append(history.history)

    # # Save the history as a pickle file
	# history_file_path = os.path.join(config.PICKLE_OUTPUT_PATH, f"history_epoch_{epoch:02d}.pkl")
	# with open(history_file_path, 'wb') as history_file:
	# 	pickle.dump(history.history, history_file)

# save the model to disk
print("[INFO] serializing network...")
model.save(config.TRAINED_MODEL_PATH)

# save the training history list as a pickle file
# all_history_file_path = os.path.join(config.PICKLE_OUTPUT_PATH, "all_history.pkl")
# with open(all_history_file_path, 'wb') as all_history_file:
#     pickle.dump(all_history, all_history_file)
    
# save the training loss and accuracy plot
# plt.style.use("ggplot")
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.savefig(config.ACCURACY_LOSS_PLOT_PATH)
