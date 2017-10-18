#vanilla rnn with sequnce input and and output
import numpy as np

class RNN(object):
    """
    layers for vanilla rnn
    """

    def __init__(self, name = ""):
        self.name = name

    def pred(self):
        t = 250
        X, Y = self.getNextBatch(1, t) #only used to extract the first word --> starting word 

        H = self.H
        preds = []; indexs = []; origin = []

        Wh = self.Wh; Wx = self.Wx; bh = self.bh
        Wy = self.Wy; by = self.by

        X = X[0]
        for i in range(1, t): #to1, to t 
            Y_hat, Z, H, A = self.rnn_forward(H, X, Wh, Wx, bh, Wy, by)
            preds.append(self.toChar[np.argmax(Y_hat[0])])
            X = np.zeros((1, self.vsize))
            X[0][self.toIndex[preds[i - 1]]] = 1
            indexs.append(np.argmax(Y_hat[0]))

        print (indexs)
        print ("".join(preds))

    def dataPre(self, filename):
        self.dataRaw = open(filename, 'r').read()
        self.p = 0
        chars = set(self.dataRaw)
        self.vsize = len(chars) 

        toIndex = {}
        for c in chars:
            toIndex[c] = len(toIndex)

        toChar = {}
        for c in toIndex:
            toChar[toIndex[c]] = c

        print (toChar); print (toIndex)

        self.toIndex = toIndex
        self.toChar = toChar

    def getNextBatch(self, n, t):
        X = np.zeros((t, n, self.vsize))
        Y = np.zeros((t, n, self.vsize))

        for i in range(n):
            x, y = self.getNextSample(t) #t, self.vsize 
            for j in range(t):
                X[j][i] = x[j]; Y[j][i] = y[j]

        return X, Y

    def getNextSample(self, t):
        start = self.p; end = start + t - 1; self.p += t 
        if (end + 1 >= len(self.dataRaw)):
            self.p = 0; start = self.p; end = start + t - 1; self.p += t

        if (end + 1 >= len(self.dataRaw)):
            raise "bad things"
        x = self.dataRaw[start : end + 1]
        y = self.dataRaw[start + 1 : end + 2]

        X = np.zeros((t, self.vsize))
        for i in range(t):
            X[i][self.toIndex[x[i]]] = 1

        Y = np.zeros_like(X)
        for i in range(t):
            Y[i][self.toIndex[y[i]]] = 1

        return X,Y

    def softmax(self, x):
        top = np.max(x, axis = 1, keepdims = True)
        x = x - top 
        x = np.exp(x) / np.exp(x).sum(axis = 1, keepdims = True)
        return x

    def cross_entropy(self, Y_hat, Y):
        return -(Y * np.log(Y_hat + 0.00000000001)).sum()

    def rnn_forward(self, H_t0, X_t1, Wh, Wx, bh, Wy, by):
        A_t1 = H_t0.dot(Wh) + X_t1.dot(Wx) + bh 
        H_t1 = np.tanh(A_t1)
        Z_t1 = H_t1.dot(Wy) + by
        Y_t1_hat = self.softmax(Z_t1)
        return Y_t1_hat, Z_t1, H_t1, A_t1 #

    def rnn_backward(self, H_t0, Y_t0, Y_t0_hat, X_t1, A_t1, dH_t1, Wh, Wx, bh, Wy, by):
        dy = Y_t0_hat - Y_t0 # 1 * y 
        dWy = H_t0.T.dot(dy)  # h * 1 1 * k = h * y
        dby = np.sum(dy, axis = 0) 

        dA_t1 = (1 - np.tanh(A_t1) * np.tanh(A_t1)) * dH_t1 # 1 * h 
        dWx = X_t1.T.dot(dA_t1) # x * n * n * h 
        dWh = H_t0.T.dot(dA_t1)
        dbh = np.sum(dA_t1, axis = 0)

        dH_t0 = dy.dot(Wy.T) + dA_t1.dot(Wh.T) # 1 * k *  k * h + 1 * h * h * h = 1 * h 
        return dH_t0, dWh, dWx, dbh, dWy, dby

    def train(self, X, Y, K = 10000000, alpha = 0.01):
        t, n, xLen = X.shape
        t, n, yLen = Y.shape
        X = np.vstack((X[0].reshape(1, n, xLen), X)) # add x[0], just make index easier to handle
        Y = np.vstack((np.zeros_like(Y[0]).reshape(1, n, yLen), Y))

        hLen = 100;
        Wh = np.random.randn(hLen, hLen);   Wh /= np.sqrt(np.prod(Wh.shape[:])) 
        Wx = np.random.randn(xLen, hLen);   Wx /= np.sqrt(np.prod(Wx.shape[:])) 
        bh = np.zeros((1, hLen)) 
        Wy = np.random.randn(hLen, yLen);   Wy /= np.sqrt(np.prod(Wy.shape[:])) 
        by = np.zeros((1, yLen))        
        self.Wh = Wh; self.Wx = Wx; self.bh = bh
        self.Wy = Wy; self.by = by

        H = np.zeros((t, n, hLen))
        A = np.zeros_like(np.vstack((H[0].reshape(1, n, hLen), H)))
        Y_hat = np.zeros_like(Y)
        Z = np.zeros_like(Y)

        for j in range(K):
            H[0] = H[t - 1]
            self.H = H[:,4,:]
            if (j % 500 == 0):
                self.pred()
            X, Y = self.getNextBatch(batch_size, 25)
            X = np.vstack((X[0].reshape(1, n, xLen), X)) # add x[0], just make index easier to handle
            Y = np.vstack((np.zeros_like(Y[0]).reshape(1, n, yLen), Y))
            dWh = np.zeros_like(Wh); dWx = np.zeros_like(Wx); dbh = np.zeros_like(bh);
            dWy = np.zeros_like(Wy); dby = np.zeros_like(by); 

            #forward:
            cost = 0.0
            for i in range(1, t): #to1, to t 
                Y_hat[i], Z[i], H[i], A[i] = self.rnn_forward(H[i - 1], X[i], Wh, Wx, bh, Wy, by) 
                cost += self.cross_entropy(Y_hat[i], Y[i])

            #backward:
            dH_next = np.zeros((n, hLen)) # or H like..
            for i in reversed(range(0, t)): #to i. 
                dH_next , dWh_ti, dWx_ti, dbh_ti, dWy_ti, dby_ti = self.rnn_backward(H[i], Y[i], Y_hat[i], X[i + 1], A[i + 1], dH_next, Wh, Wx, bh, Wy, by)

                dWh += dWh_ti; dWx += dWx_ti; dbh += dbh_ti
                dWy += dWy_ti; dby += dby_ti

            #gradient update. -- done! : )
            Wh -= alpha * dWh
            Wx -= alpha * dWx
            bh -= alpha * dbh
            Wy -= alpha * dWy
            by -= alpha * dby

            self.Wh = Wh; self.Wx = Wx; self.bh = bh
            self.Wy = Wy; self.by = by
            print ("iteration : ", j, " cost : ", cost)

        return Wh, Wx, bh, Wy, by 

batch_size = 5
rnn = RNN()
rnn.dataPre('./testingArtical2')
X, Y = rnn.getNextBatch(batch_size, 25)
rnn.train(X, Y)
