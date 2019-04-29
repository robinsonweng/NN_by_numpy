import numpy as np 
import matplotlib.pyplot as plt


def np_data():
    with open("dataset/iris.txt", "r", encoding="utf-8") as f:
        data = [line.split(",") for line in f]

    training_data = data[0:4] + data[50:54] 
    output = np.array([[0],
                       [0],
                       [0],
                       [0],
                       [1],
                       [1],
                       [1],
                       [1]])
        
    trainingdata = [line[0:4] for line in training_data]
    np_training = np.asarray(trainingdata)
    float_traning = np_training.astype(np.float32)
    np_output = np.asarray(output)
    int_output = np_output.astype(int)

    return float_traning, int_output

def sigmoid(x, De=False):
    if not De:
        return 1.0/(1 + np.exp(-x))
    else:
        out = sigmoid(x)
        return out *(1 - out)

def FP_BP_wB(training_rate, traning_times):
    """Training NN with bias"""
    #init data
    eta = training_rate
    training, output = np_data()
    #init weight
    w1 = np.random.random((4, 4)) - 1 #4x4
    w1_0 = np.ones([1, 4])
    w1_ = np.vstack([w1_0, w1])

    w2 = np.random.random([1, 4]) - 1 #1x4
    w2_0 = np.ones([1, 1])  
    w2_ = np.hstack([w2_0, w2])

    #init bias
    bias1 = np.random.random([8, 1])
    bias2 = np.random.random([8, 1])

    #FP
    Layers = []
    losses = []
    for times in range(traning_times):
        l0 = training
        l0_bias = np.hstack([bias1, l0]) #8x5
        l1 = sigmoid(np.dot(l0_bias, w1_)) #8x4
        l2_bias = np.hstack([bias2, l1]) #8x5
        l2 = sigmoid(np.dot(l2_bias, w2_.T)) # 8x1
        
        #error
        error = output - l2
        #MSE
        E = 1/2 * pow(error, 2)
        
        #BP
        delta = error * sigmoid(np.dot(l2_bias, w2_.T), De=True) # 8x1
        w2_gradient = delta.T.dot(l1) #1x4
        w1_gradient = (np.dot(delta, w2) * sigmoid(np.dot(l0_bias, w1_), De=True)).T.dot(l0) #4x4
        
        bias2_graient = error * sigmoid(np.dot(l2_bias, w2_.T)) #8x1
        bias1_graient = sigmoid(np.dot(l0_bias, w1_), De=True).dot(w2.T) * delta #8x4
        #update
        w2 += eta * w2_gradient
        w1 += eta * w1_gradient
        bias2 += bias2_graient
        bias1 += bias1_graient        
        if times % 1000 == 0:
            MSE = np.mean(np.abs(E))
            losses.append(MSE)
            Layers.append(np.mean(l2))
            print(MSE)

    return w2, w1, bias2, bias1

def test(FB_BP):
    """testing NN"""
    #init data
    testing = np.array([[5.1,3.8,1.9,0.4], #sentosa 
                           [6.9,3.1,4.9,1.5], #versicolor               
                           [4.8,3.0,1.4,0.3], #sentosa
                           [6.9,3.1,4.9,1.5], #versicolor
                           [4.9,3.6,1.4,0.1], #sentosa
                           [5.5,4.2,1.4,0.2], #sentosa
                           [4.4,3.2,1.3,0.2], #sentosa
                           [5.6,2.9,3.6,1.3]])#versicolor

    Y = np.array([[0],
                  [1],
                  [0],
                  [1],
                  [0],
                  [1],
                  [0],
                  [1]])

    #init weihght & bias
    weight2, weight1, bias2, bias1 = FB_BP(0.9, 2000)
    
    w1_ones = np.ones([1, 4])
    w1 = np.vstack([w1_ones, weight1])
    
    w2_ones = np.ones([1, 1])
    w2 = np.hstack([w2_ones, weight2])
    
    l0 = testing
    l0_bias = np.hstack([bias1, l0])
    l1 = sigmoid(np.dot(l0_bias, w1))
    l1_bias = np.hstack([bias2, l1])
    l2 = sigmoid(np.dot(l1_bias, w2.T))

    return l2

NN = test(FP_BP_wB)
print(NN)
