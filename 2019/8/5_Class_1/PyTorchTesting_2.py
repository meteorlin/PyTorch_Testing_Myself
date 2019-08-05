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
# 在pytorch中，如果需要后面对某一个参数进行自动计算梯度，
# 则需要在声明的时候添加上 requires_grad=True

# 在之前的pytorch版本中，requires_grad 默认为 True，但是由于默认是 True，
# 所以会占用一定的内存，故新版本使用该标记来减少系统内存的消耗。
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

# 定义学习率（learning_rate）
learning_rate = 1e-6

# 神经网络训练
for it in range(500):
    # 前向传播
        # h = x.mm(w1) # 定义中间层转化函数
        # h_relu = h.clamp(min = 0) # 写激活函数
        # y_pred = h_relu.mm(w2) # 定义最终输出的预测函数
    # 将上述式子精简成一个式子
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)

    # 计算损失损失函数L2（均方误差）
    # 注意下面这个loss不能添加item()了，因为后面自动计算梯度的backward()需要使用到计算图，
    # 而item()会直接将 1行1列 的Tensor变成一个scalar
    loss = (y_pred - y).pow(2).sum() # computation graph （计算图，用于后面的自动计算各参数梯度）
    print(it, loss.item())

    # 反向传播（也就是计算梯度下降的过程）
    # compute the gradient
    # 使用backward()自动计算loss中相关参数的梯度
    loss.backward()

    # update weight of w1 and w2
    # 这里可以直接调用pytorch的自动计算梯度（前提是在前面我们已经对w1 和w2 声明了requires_grad = True）
    # 需要注意的是，在pytorch中，所有Tensor的计算都是一张计算图（computation graph）
    # 会占用一定的内存，所以在更新下面的w1 w2 参数时，需要声明不再申请空间来存储w1 w2 的 grad
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 注意在下一次计算loss梯度之前需要对对应参数的grad清零，否则梯度值会一直累积
        w1.grad.zero_()
        w2.grad.zero_()