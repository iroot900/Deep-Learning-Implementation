import layers
import numpy as np


# input conv1 sub2 conv3 sub4 conv5 full6 output
# model['w1', 'b1', ...6, 'wo', 'bo']

def forward(data, model) : 
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
    return out #soft max is linear so ok

def getTestRightCount(data, model):
    target_test = data[:, -1]
    target_test = np.array(target_test)
    target_test.resize(data.shape[0], 1)
    data = data[: ,0:28 * 28]
    data = np.array(data)
    data.resize(data.shape[0], 1, 28, 28)
    Y_test = forward(data, model) 
    Y_arg_max_test = Y_test.argmax(axis = 1)
    return layers.classification_count(Y_arg_max_test, target_test)

def train(data, i, model, alpha = 0.0001): 
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

    n = data.shape[0]
    t = 5
    start = i * t; start = start % n; end = start + t
    data = data[start : end, ]

    #data to data and y. label and model 
    y = np.array(data[:,-1]); y.resize(t, 1)
    label = np.zeros((t, 10))
    for i in range(t):
      label[i][int(y[i])] = 1

    data = np.array(data[:,0:28 * 28])
    data.resize(t, 1, 28, 28)
    data = data * 1.01

    #drop out rate
    #drop out implementation is ugly here, should move to layer as an operation and transparent to convnets
    p = 0.95

    #forward pass
    h1_pre = layers.conv_forward(data, model[w1], model[b1])
    h1 = layers.ReLu_forward(h1_pre)
    #print (h1[0][0])

    h2 = layers.max_pool(h1, 2)
    U2 = (np.random.rand(*h2.shape) < p) / p 
    h2 *= U2 # drop!

    h3_pre = layers.conv_forward(h2, model[w3], model[b3])
    h3 = layers.ReLu_forward(h3_pre)

    h4 = layers.max_pool(h3, 2)
    U4 = (np.random.rand(*h4.shape) < p) / p 
    h4 *= U4 # drop!

    h5_pre = layers.conv_forward(h4, model[w5], model[b5])
    h5 = layers.ReLu_forward(h5_pre)

    U5 = (np.random.rand(*h5.shape) < p) / p
    h5 *= U5 # drop!

    h6 = layers.full_forward(h5, model[w6], model[b6]) 
    U6 = (np.random.rand(*h6.shape) < p) / p 
    h6 *= U6 # drop!

    out = layers.full_forward(h6, model[wo], model[bo])
    y_hat = layers.softmax(out)

    y_hat_arg = np.argmax(y_hat, axis = 1)
    dout = (y_hat - label)
    cost = layers.cost(y_hat, label)
    rate = layers.classification_rate(y, y_hat_arg)

    print ("------")
    print ("gradient updates : ");
    print ("cost : ", cost) 
    print ("rate : ", rate) 


    dout_h6, dwo_gradient, dbo_gradient = layers.full_backward(dout, h6, model[wo], model[bo])
    dout_h6 *= U6 

    dout_h5, dw6_gradient, db6_gradient = layers.full_backward(dout_h6, h5, model[w6], model[b6])
    dout_h5 *= U5 

    dout_h4, dw5_gradient, db5_gradient = layers.conv_backward(layers.ReLu_backward(h5_pre, dout_h5), h4, model[w5], model[b5])
    dout_h4 *= U4 

    dout_h3 = layers.max_pool_back(h3, dout_h4, 2)

    dout_h2, dw3_gradient, db3_gradient = layers.conv_backward(layers.ReLu_backward(h3_pre, dout_h3), h2, model[w3], model[b3])
    dout_h2 *= U2 
   
    dout_h1 = layers.max_pool_back(h1, dout_h2, 2)

    d_data, dw1_gradient, db1_gradient = layers.conv_backward(layers.ReLu_backward(h1_pre, dout_h1), data, model[w1], model[b1])


    gradients = {}
    gradients[wo] = dwo_gradient;
    gradients[bo] = dbo_gradient;

    gradients[w6] = dw6_gradient;
    gradients[b6] = db6_gradient;

    gradients[w5] = dw5_gradient;
    gradients[b5] = db5_gradient;

    gradients[w3] = dw3_gradient;
    gradients[b3] = db3_gradient;

    gradients[w1] = dw1_gradient;
    gradients[b1] = db1_gradient;

    return [gradients, cost, rate] 

def modelPre():
    model = {}
    model["w1"] = np.random.randn(6, 1, 3, 3)
    model["b1"] = np.random.randn(6)

    model["w3"] = np.random.randn(16, 6, 3, 3)
    model["b3"] = np.random.randn(16)

    model["w5"] = np.random.randn(120, 16, 3, 3) # 120 * 7 * 7
    model["b5"] = np.random.randn(120)

    model["w6"] = np.random.randn(120 * 7 * 7, 84)
    model["b6"] = np.random.randn(84)


    model["wo"] = np.random.randn(84, K)
    model["bo"] = np.random.randn(K)

    for weight in model:
        model[weight] /= np.sqrt(np.prod(model[weight].shape[:]))

    return model

K = 10

