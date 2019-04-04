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

def FP_BP_nB(training_rate, training_times):
    """Training NN with out bias"""
    #init data
    eta = training_rate
    training, output = np_data()
    #init weight
    w1 = np.random.random((4, 4))
    w2 = np.random.random((1, 4)) 
    

    for times in range(training_times):
        #FP
        l0 = training # 8X4
        l1 = sigmoid(np.dot(l0, w1.T)) #8X4
        l2 = sigmoid(np.dot(l1, w2.T)) #8X1
        #errot
        error = output - l2#8X1 
        #MSE
        E = 1/2 * pow(error, 2)
        if times % 1000 == 0:
            print(np.mean(np.abs(E)))

        #BP
        delta = error * sigmoid(np.dot(l1, w2.T), De=True) #8X1
        delta_w2 = delta.T.dot(l1) #1X4
        delta_w1 = (np.dot(delta, w2) * sigmoid(np.dot(l0, w1.T))).T.dot(l0) #4X4
        #update
        w2 += eta * delta_w2
        w1 += eta * delta_w1
        
    return w1, w2

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
    weight1, weight2 = FB_BP(0.1, 7000)
    
    l0 = testing
    l1 = sigmoid(np.dot(l0, weight1.T))
    l2 = sigmoid(np.dot(l1, weight2.T))
   
    return l2

NN = test(FP_BP_nB)
print(NN)