# coding=utf-8
import numpy as np
#激活函数，sigmoid
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
#四个样本，每个样本有三个特征
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
print X.shape
#类别标签
y = np.array([[0],
            [1],
            [1],
            [0]])
print y.shape
np.random.seed(1)

# randomly initialize our weights with mean 0
w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1
print w0
print w1
print w0.shape
print w1.shape
#迭代训练
for j in xrange(60000):

    # 计算前项传播
    l0 = X
    l1 = nonlin(np.dot(l0,w0))
    l2 = nonlin(np.dot(l1,w1))

    #计算反向传播
    #计算L2损失
    l2_error = y - l2

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    l2_delta = l2_error*nonlin(l2,deriv=True)

    l1_error = l2_delta.dot(w1.T)

    l1_delta = l1_error * nonlin(l1,deriv=True)

    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)