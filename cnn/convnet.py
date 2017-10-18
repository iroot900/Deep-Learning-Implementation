import layers
import numpy as np


# input conv1 sub2 conv3 sub4 conv5 full6 output
# model['w1', 'b1', ...6, 'wo', 'bo']

def train(data, label, y, model, alpha = 0.001): 
    """
    Input:
        data  : (N, C, H, W)
        label : (N, K)
        y     : (N, )
    Output:
        cost : 
        rate :
    """
    w1 = "w1"; b1 = "b1"
    w3 = "w3"; b3 = "b3"
    w5 = "w5"; b5 = "b5"
    w6 = "w6"; b6 = "b6"
    wo = "wo"; bo = "bo"

    #forward pass
    h1_pre = layers.conv_forward(data, model[w1], model[b1])
    h1 = layers.ReLu_forward(h1_pre)
    #print (h1[0][0])

    h2 = layers.max_pool(h1, 2)

    h3_pre = layers.conv_forward(h2, model[w3], model[b3])
    h3 = layers.ReLu_forward(h3_pre)

    h4 = layers.max_pool(h3, 2)

    h5_pre = layers.conv_forward(h4, model[w5], model[b5])
    h5 = layers.ReLu_forward(h5_pre)

    h6 = layers.full_forward(h5, model[w6], model[b6]) 

    out = layers.full_forward(h6, model[wo], model[bo]) #after this we need softmax 
    y_hat = layers.softmax(out)

    y_hat_arg = np.argmax(y_hat, axis = 1)
    dout = (y_hat - y)
    cost = layers.cost(y_hat, y)
    rate = layers.classification_rate(label, y_hat_arg)

    #gd
    print ("------")
    print (y_hat)
    print ("gradient updates : ");
    print ("------")

    dout_h6, dwo_gradient, dbo_gradient = layers.full_backward(dout, h6, model[wo], model[bo])
    model[wo] =  model[wo] - alpha * dwo_gradient
    model[bo] =  model[bo] - alpha * dbo_gradient

    dout_h5, dw6_gradient, db6_gradient = layers.full_backward(dout_h6, h5, model[w6], model[b6])
    model[w6] =  model[w6] - alpha * dw6_gradient
    model[b6] =  model[b6] - alpha * db6_gradient

    dout_h4, dw5_gradient, db5_gradient = layers.conv_backward(layers.ReLu_backward(h5_pre, dout_h5), h4, model[w5], model[b5])
    model[w5] =  model[w5] - alpha * dw5_gradient
    model[b5] =  model[b5] - alpha * db5_gradient

    dout_h3 = layers.max_pool_back(h3, dout_h4, 2)

    dout_h2, dw3_gradient, db3_gradient = layers.conv_backward(layers.ReLu_backward(h3_pre, dout_h3), h2, model[w3], model[b3])
    model[w3] =  model[w3] - alpha * dw3_gradient
    model[b3] =  model[b3] - alpha * db3_gradient
   
    dout_h1 = layers.max_pool_back(h1, dout_h2, 2)

    d_data, dw1_gradient, db1_gradient = layers.conv_backward(layers.ReLu_backward(h1_pre, dout_h1), data, model[w1], model[b1])
    model[w1] =  model[w1] - alpha * dw1_gradient
    model[b1] =  model[b1] - alpha * db1_gradient

    return cost, rate

def modelPre():
    model = {}
    model["w1"] = np.random.randn(6, 3, 3, 3)
    model["b1"] = np.random.randn(6)

    model["w3"] = np.random.randn(16, 6, 3, 3)
    model["b3"] = np.random.randn(16)

    model["w5"] = np.random.randn(120, 16, 3, 3) # 120 * 7 * 7
    model["b5"] = np.random.randn(120)

    model["w6"] = np.random.randn(120 * 4 * 4, 84) # n /2 / 2
    model["b6"] = np.random.randn(84)


    model["wo"] = np.random.randn(84, K)
    model["bo"] = np.random.randn(K)

    for weight in model:
        model[weight] /= np.sqrt(np.prod(model[weight].shape[:]))

    return model

K = 2
N = 100
data = np.random.randn(N, 3, 16, 16)
label = np.zeros(N)
y = np.zeros((N, K))

data[0:50] += 0.5 
data[50:] += -0.5
label[0:50] = 1
label[50:] = 0

for i in range(N):
    y[i][int(label[i])] = 1


#pre-process
data = data / np.std(data, axis = 0)

model = modelPre()
for i in range (100):
    cost, rate = train(data, label, y, model)
    print ("iteration :", i)
    print (" cost : ", cost)
    print (" rate : ", rate)
