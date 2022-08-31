from tensorflow.keras.datasets import mnist
((trainX, trainY), (testX, testY)) = mnist.load_data()
print(testY[0])
