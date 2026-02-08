"""
This wont work because I didnt want to implement back propagation in this right now... just switching to PyTorch
"""

import numpy as np

x = []
y = []

X_train, X_test = x[:len(x)*9//10]/255, x[len(x)*9//10:]/255
y_train, y_test = y[:len(x)*9//10], y[len(x)*9//10:]

W1 = 0.01*np.random.rand(42, 128)
b1 = np.zeros(128, )
W2 = 0.01*np.random.rand(128, 10)
b2 = np.zeros(10, )

def relu(Z):
    Z = np.maximum(0, Z)
    return Z

def forward_pass(x, W1, W2, b1, b2):
    Zi = x@W1 + b1
    Z = relu(Zi)
    Z = Z@W2 + b2
    return Z, Zi

def softmax(Z):
    P = []
    for Zi in Z:
        Za = Zi - max(Zi)
        Pi = np.exp(Za)/sum(np.exp(Za))
        P.append(Pi)
    return np.array(P)

def CrossEntropyLoss(P, y):
    c = 0
    L = []
    for i in P:
        loss = -np.log(i[int(y[c])]+0.000001)
        c+=1
        L.append(loss)
    mean_loss = sum(L) / len(L)
    return mean_loss

"""
For epoch in range
do the forward_pass
calc the loss
optimizer zerograd
loss backward
optimzer step
"""

def backward(P,y,b_s):
    dZ = P.copy()
    for a in range(0, len(P)):
        dZ[a, int(y[a])] -= 1
    dZ /= b_s
    return dZ

for epoch in range(1,101):
    print(f"=========================== {epoch}/10 Started ===========================")
    X_sample = X_train[:]
    y_sample = y_train[:]
    Z, Zi = forward_pass(X_sample, W1, W2, b1, b2)
    P = softmax(Z)
    L = CrossEntropyLoss(P, y_sample)
    dZ = backward(P, y_sample, len(X_train))
    print(f"Loss for {epoch}: {L}")
    print(f"=========================== {epoch}/10 Completed ===========================\n\n\n")

np.savetxt("weights1.txt", W1)
np.savetxt("weights2.txt", W2)
np.savetxt("bias.txt", b1)

