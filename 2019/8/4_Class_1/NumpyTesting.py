import numpy as np

'''
    由于numpy是一个矩阵运算包，所以里面没有一些集成的神经网络的快捷包
    使用numpy进行书写神经网络代码有利于我们真正测试自己是否学习到了神经网络的具体运算步骤和数学原理
    这个程序要完全使用numpy来实现
    # 前向传播（forward pass）
        # 这个神经网络的架构为
            # 一个全连接的ReLU网络
            # 一个隐藏层
            # 目标函数没有偏移量（bias）
        # 所要实现的目标和使用的损失函数
            # 用于使用x预测y
            # 使用L2损失函数
    # 损失函数的定义（loss）
    # 反向传播（backward pass）
'''

# 超变量定义（hypeparameters）
# 输入个数 N = 64
# 输入维数 D_in = 1000
# 中间隐藏层维数 H = 100
# 输出维数 D_out = 10
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# w1是y{hat} = w1 * x 中，能把x从1000维转化到100维的矩阵,w2同理。
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# 定义学习率（learning_rate）
learning_rate = 1e-6

# 神经网络训练
for it in range(500):
    # 前向传播
    h = x.dot(w1) # 定义中间层转化函数
    h_relu = np.maximum(h, 0) # 写激活函数
    y_pred = h_relu.dot(w2) # 定义最终输出的预测函数

    # 计算损失损失函数L2（均方误差）
    loss = np.square(y_pred - y).sum()
    print(it, loss)

    # 反向传播（也就是计算梯度下降的过程）
        # numpy需要手动定义每一步的梯度计算式子
        # 最终是对loss中的参数进行求导，但是由于loss和w1没有直接关系
        # 所以现在如果想要求 d loss / d w1，那就需要使用一系列的链式求导法则将各个阶段的导数值乘起来
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y) # 这个是直接对loss对y_pred进行求导
    grad_w2 = h_relu.T.dot(grad_y_pred)
        # 鉴于ReLU的特殊结构，可以使用下面的式子进行定义
        # 也就是说，在计算ReLU的梯度时，[0,∞)部分导数是线性的
        # 而在 h<0 部分h的导数是0
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy() # 备份一份grad_h_relu.
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    # update weight of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2