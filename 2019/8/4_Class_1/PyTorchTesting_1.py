import torch

'''
    PyTorch的Tensor与numpy的ndarray和相似，但是最大的区别是PyTorch Tensor可以在CPU或GPU上运算。
    如果想在GPU上运算，需要把Tensor换成CUDA类型。
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

'''
    要想在numpy和pytorch之间转换，需要进行以下步骤：
        # np.random.randn -> torch.randn
        # dot -> mm （内积）
        # np.maximum(h, 0) -> torch.clamp(min = 0)（ReLU激活函数）
            # clamp(min = x,max = y)，就是将输入的值夹在[x, y]之间
        # loss = np.square(y_pred - y).sum() -> loss = (y_pred - y).pow(2).sum().item() （均方误差）
            # 在最后需要加上一个item()，因为 (y_pred - y).pow(2).sum() 计算得到的是一个Tensor（1行1列）
            # 所以需要加上item()，将该 1行1列 的Tensor转换成Python的数字类型（number）
        # T -> t() （矩阵转置）
        # copy() -> clone() （矩阵备份）
'''

# 超变量定义（hypeparameters）
# 输入个数 N = 64
# 输入维数 D_in = 1000
# 中间隐藏层维数 H = 100
# 输出维数 D_out = 10
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# w1是y{hat} = w1 * x 中，能把x从1000维转化到100维的矩阵,w2同理。
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

# 定义学习率（learning_rate）
learning_rate = 1e-6

# 神经网络训练
for it in range(500):
    # 前向传播
    h = x.mm(w1) # 定义中间层转化函数
    h_relu = h.clamp(min = 0) # 写激活函数
    y_pred = h_relu.mm(w2) # 定义最终输出的预测函数

    # 计算损失损失函数L2（均方误差）
    loss = (y_pred - y).pow(2).sum().item()
    print(it, loss)

    # 反向传播（也就是计算梯度下降的过程）
        # numpy需要手动定义每一步的梯度计算式子
        # 最终是对loss中的参数进行求导，但是由于loss和w1没有直接关系
        # 所以现在如果想要求 d loss / d w1，那就需要使用一系列的链式求导法则将各个阶段的导数值乘起来
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y) # 这个是直接对loss对y_pred进行求导
    grad_w2 = h_relu.t().mm(grad_y_pred)
        # 鉴于ReLU的特殊结构，可以使用下面的式子进行定义
        # 也就是说，在计算ReLU的梯度时，[0,∞)部分导数是线性的
        # 而在 h<0 部分h的导数是0
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone() # 备份一份grad_h_relu
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    # update weight of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2