import numpy as np
import os, sys, time
# Surpress TF warnings (especially those relating to AVX512 (CPU instructions set) and RTCores (Nvidia GPU 2000 Series or greater)). None of those are available on my PCs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     
import matplotlib.pyplot as plt
from keras.utils import plot_model
from tensorflow import config, sysconfig
from tensorflow.keras import datasets, layers, models, utils

print("################## GENERAL INFO ##################")   # General info about the system. Allows to check if the GPU is being used or not
GPUs = config.list_physical_devices('GPU')
if len(GPUs) > 0:
    print("GPU:", config.experimental.get_device_details(GPUs[0]).get('device_name'))
    sys_details = sysconfig.get_build_info()
    cuda = sys_details["cuda_version"]
    cudnn = sys_details["cudnn_version"]
    print("CUDA Version:", cuda)
    print("cudNN Version:", cudnn)
else:
    print("No GPU available, using CPU instead")
print("################## GENERAL INFO ##################\n")

SALVAR_PARA_MEMORIA = True   # To save the data that is going to be used in the report. Not meant to be used in the final version or by third parties.

mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # X = 28x28 images, Y = labels  
x_train = x_train / 255.0   # Normalize the data
x_test = x_test / 255.0

# Save the first 5 images of the training set to a file. To be used in the report just once
"""
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.savefig("results/first_5_images.png")
"""

opcion = -1
while opcion < 0 or opcion > 7:
    print("Choose a model to train (numeric index):")
    print("\t1. Simple perceptron")
    print("\t2. Multilayer neural network, 256 neurons in the hidden layer and softmax output layer")
    print("\t3. Convolutional neural network trained with stochastic gradient descent")
    print("\t4. LeNet-5 Convolutional neural network (trained with Adam)")
    print("\t5. Another convolutional neural network (modified from Keras' documentation)")
    print("\t6. Another CNN with 4 convolutional layers (also batch normalization and dropouts, simplified from Kaggle's notebook)")
    print("\t7. CNN from Kaggle's notebook (without any modification)")
    print("\t0. Salir")
    opcion = int(input("Index: "))
    print("")

if opcion >= 3:
    y_train = utils.to_categorical(y_train, num_classes=10)   # One-hot encoding for the labels
    y_test = utils.to_categorical(y_test, num_classes=10)

if opcion == 0:
    print("Saliendo...")
    sys.exit()

elif opcion == 1:
    # Simple perceptron, one neuron per class
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(10, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=10, batch_size=128)  # Same accuracy as no_batch_size for test data but ~3.4 times faster
    elapsed_time = time.time() - start_time                           # 20 epochs with no_bacth_size gives slightly better results

elif opcion == 2:
    # Multilayer neural network, 256 neurons in the hidden layer and softmax output layer
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),   
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),                              # With dropout until 0.5 accuracy is almost the same; 0.2 gives better results
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128)  # Same accuracy as no_batch_size for test data but ~3.4 times faster
    elapsed_time = time.time() - start_time # 10 128

elif opcion == 3:
    # Convolutional neural network trained with stochastic gradient descent (SGD)
    # Different from LeNet-5 as filters' size are 32-multiple. Fully connected layers are also different
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),    
        layers.MaxPooling2D((2,2)),            
        layers.Conv2D(64, (3,3), activation='relu'),    # Extra convolutional layer
        layers.Flatten(),
        layers.Dropout(0.5),                            # LeNet-5 doesn't have dropout
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=10, batch_size=32) # Tested with lots of combinations of batch_size and epochs. Decide to use epochs=10, batch_size=32 and dropout=0.5
    elapsed_time = time.time() - start_time               # Best results with epochs=20, batch_size=32 and no dropout. Only 0,0002 more accuracy and takes ~2.6 times longer

elif opcion == 4:
    # Lenet-5. Convolutional neural network trained with Adam insted of the original SGD
    model = models.Sequential([
        layers.Conv2D(6, (5,5), padding='same', activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # LeNet-5 uses SGD, but Adam is better. SGD is tested in previous model

    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128)  # Using batch_size=128 is almost mandatory as execution times are too long otherwise
    elapsed_time = time.time() - start_time

elif opcion == 5:
    # Another convolutional neural network, modified from examples of Keras' documentation
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (5,5), activation='relu'),    # Extra convolutional layer. Tested with and without it, but better results are obtained with it.
        layers.AvgPool2D(pool_size=(2, 2)), # Average pooling instead of max pooling; gives better results
        layers.Dropout(0.2),    # 0.2 seems to gives better results. Almost the same as original 0.25
        layers.Flatten(),       
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),    # Tested from 0.2 to 0.5; 0.4 seems go give better results. Almost the same as original 0.5
        layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128)  # Best result with epochs=15, batch_size=128. Usign epochs=20 is 0.02% less accurate but quite faster to train.
    elapsed_time = time.time() - start_time                           # Using batch_size=128 is almost mandatory as execution times are too long otherwise

elif opcion == 6:
    # Convolutional neural network "based" on the Keras' documentation example, but we add bach normalization after each convolutional layer, 
    # more convolutional layers with different filters' size, dropout after each "stage" and avg pooling instead of max pooling before the fully connected layers
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (5,5), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.AvgPool2D(pool_size=(2, 2)), 

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),    
        layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128)  # Using batch_size=128 is almost mandatory as execution times are too long otherwise
    elapsed_time = time.time() - start_time                          # Best result with epochs=30, batch_size=128. Only 0.01% more accuracy and quite longer than epochs=20

elif opcion == 7:
    # Kaggle's notebook without any modification. All credits to the author.
    model = models.Sequential([
        layers.Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32,kernel_size=3,activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Conv2D(64,kernel_size=3,activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64,kernel_size=3,activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=35)  # Doesn't mention batch_size, so we use the default value
    elapsed_time = time.time() - start_time     

else:
    print("Opción incorrecta. Saliendo...")  # This should never happen
    sys.exit()

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\nPrecisión en entrenamiento:", round(train_acc, 5), "\t Precisión en test: ", round(test_acc,5))
print("Pérdida en entrenamiento:", round(train_loss,5),  "\t Pérdida en test: ", round(test_loss,5))
print("Tiempo de entrenamiento:", round(elapsed_time,5), "segundos")  # Time in seconds

if SALVAR_PARA_MEMORIA:
    # Create the directory for the results if it doesn't exist. Files names and their organization will be made by hand.
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/models'):
        os.makedirs('results/models')
    if not os.path.exists('results/predictions'):
        os.makedirs('results/predictions')
    if opcion >=3:
        y_test = np.argmax(y_test, axis=1)  # Convert one-hot encoding to single integer label
    np.savetxt(f"results/predictions/opcion_{opcion}.txt", y_test, fmt="%i", newline='')  # Print the test labels in a single line (export to file)
    plot_model(model, to_file=f"results/models/model_opcion_{opcion}.png", show_shapes=True, show_layer_names=True)
    model.save(f"results/models/model_opcion_{opcion}.h5") # Save the model to a file
    model.summary()  # Print the model summary. Check the number of parameters for the report.
    if opcion == 6:
        y_test_labels = model.predict(x_test) 
        error_idx = np.where(np.argmax(y_test_labels, -1) != y_test)
        # Plot 30 images
        plt.figure(figsize=(20, 20))
        for i in range(30):
            plt.subplot(5, 6, i + 1)
            plt.imshow(x_test[error_idx[0][i]].reshape(28,28), cmap=plt.cm.binary)
        plt.savefig(f"results/error_images_dificiles.png")
