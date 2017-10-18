import numpy as np

def cost(Y_hat, Y): 
        return -(Y * np.log(Y_hat + 0.00000000001)).sum()

def classification_rate(label, Y_hat_arg):
    n = len(label)
    count = 0 
    for i in range(n):
        if (label[i] == Y_hat_arg[i]):
            count += 1
    return count / n 

def classification_count(label, Y_hat_arg):
    n = len(label)
    count = 0 
    for i in range(n):
        if (label[i] == Y_hat_arg[i]):
            count += 1
    return count 

def softmax(x):
    top = np.max(x, axis = 1, keepdims = True)
    x = x - top 
    x = np.exp(x) / np.exp(x).sum(axis = 1, keepdims = True)
    return x

def full_forward(x, w, b):
    print ("full connected forward")
    """
    Input:
        - x : sample data (N, C, H, W)
        - w : weights (C * H * W, M)
        - b : bias (M, )
    Output:
        - out : (N, M)
    """
    N = x.shape[0]
    length = np.prod(x.shape[1:])
    x_in_rows = x.reshape(N, length)
    out = x_in_rows.dot(w) + b
    return out

def full_backward(dout, x, w, b):
    print ("full connected backward")
    """
    Input: 
    Output:
        - dx (N, C, H, W)  /dx = delta * w
        - dw (C* H * W, M) /  dw = x * delta
        - db (M, )
    """
    N = x.shape[0]
    length = np.prod(x.shape[1:])
    
    dx_in_rows = dout.dot(w.T)
    dx = dx_in_rows.reshape(x.shape)

    db = np.sum(dout, axis = 0)
    
    x_in_rows = x.reshape(N, length)
    dw = x_in_rows.T.dot(dout)
    #dw = dw.reshape(w.shape)

    return dx, dw, db


def conv_forward(x, w, b):
    print ("conv connected forward")
    """
    Notes: to focus one propagation for now. padding to same as input size 
           stride = 1, padding (H + padding * 2 - HH + 1) = H
    Input:
	- x: sample data (N, C, H,  W)
	- w: filters     (F, C, HH, WW)
	- b: biases 	 (F, )

    Output:
	- Out: activation map (N, F, H', W') [H' = H, W' = W] -- we pad it 
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Ho = H; Wo = W
    pad_h = int((HH - 1) / 2)
    pad_w = int((WW - 1) / 2)

    x_padded = np.pad(x, [(0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)], 'constant')

    out = np.zeros((N, F, Ho, Wo));

    for i in range(N):
        for j in range(F):
            for m in range(Ho):
                for n in range(Wo):
                    hs = m * 1; ws = n * 1
                    window = x_padded[i, :, hs:hs+HH, ws:ws+WW]
                    out[i][j][m][n] = np.sum(window * w[j]) + b[j]
    return out

def conv_backward(dout, x, w, b):
    print ("conv connected backward")
    """
    Inputs:
	- dout: delta of output
    Returns a tuple of:
	- dx
	- dw
	- db
    """
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Ho = H; Wo = W
    pad_h = int((HH - 1) / 2)
    pad_w = int((WW - 1) / 2)

    x_padded  = np.pad(x,  [(0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)], 'constant')
    dx_padded = np.pad(dx, [(0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)], 'constant')


    for i in range(N):
        for j in range(F):
            for m in range(Ho):
                for n in range(Wo):
                    hs = m * 1; ws = n * 1
                    window    =  x_padded[i, :, hs:hs+HH, ws:ws+WW]
                    dx_window = dx_padded[i, :, hs:hs+HH, ws:ws+WW]
                    dw[j] += window * dout[i][j][m][n] ## dw = x * dout
                    db[j] += dout[i][j][m][n]
                    
                    #think one window -->one output a time.
                    # window * w = out.   
                    #gradient of dx.    dx = dout * w
                    dx_window += w[j] * dout[i][j][m][n] 

    dx = dx_padded[:, :, pad_h:pad_h+H, pad_w:pad_w+W]
    return dx, dw, db

def ReLu_forward(x):
    print ("relu for")
    return x * (x > 0)

def ReLu_backward(x, dout):
    print ("relu back")
    return dout * (x >= 0)

def max_pool(x, size): #down sampling
    print ("max pool")
    N, C, H, W = x.shape
    Ho = int(H / size); Wo = int(W / size)

    out = np.zeros((N, C, Ho, Wo))
    for i in range(N):
        for j in range(C):
            for m in range(Ho):
                for n in range(Wo):
                    hs = m * size; ws = n * size
                    window = x[i, j, hs:hs+size, ws:ws+size]
                    out[i][j][n][m] = np.max(window)
    return out

def max_pool_back(x, dout, size):
    print ("max pool back")
    N, C, H, W = x.shape
    Ho = int(H / size); Wo = int(W / size)

    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(C):
            for m in range(Ho):
                for n in range(Wo):
                    hs = m * size; ws = n * size
                    window    =  x[i, j, hs:hs+size, ws:ws+size]
                    dx_window = dx[i, j, hs:hs+size, ws:ws+size]
                    dx_window = dout[i][j][n][m] * (window == np.max(window))
    return dx


