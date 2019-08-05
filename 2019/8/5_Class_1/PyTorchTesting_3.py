import torch.nn as nn

'''
    使用PyTorch.nn来构建网络，使用PyTorch autograd来构建计算图和计算gradients，
    而后PyTorch 会协助我们自动计算gradient。
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

# 定义网络时，不需要单独定义w1 w2等需要学习的参数
# w1 = torch.randn(D_in, H, requires_grad=True)
# w2 = torch.randn(H, D_out, requires_grad=True)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H), # w1 * x + b1，默认bias是Ture
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# 定义学习率（learning_rate）
learning_rate = 1e-6

# 可以直接在进入神经网络训练之前定义好loss公式
loss_fn = nn.MSEloss(reduction='sum') #（均方误差L2），下面的loss也需要同步修改

# 神经网络训练
for it in range(500):
    # 前向传播
    # 将上述式子精简成一个式子
        # y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    # 采用了model之后就可以直接写如下式子
    y_pred = model(x) # 这一步直接进行前向传播操作

    # 计算损失损失函数L2（均方误差）
    loss = loss_fn(y_pred, y)
    print(it, loss.item())

    # 反向传播（也就是计算梯度下降的过程）
    # clear the memory of gradient
    # 在计算梯度之前需要把前面累积的清零
    model.zero_grad()
    # compute the gradient
    # 使用backward()自动计算loss中相关参数的梯度
    loss.backward()

    # update weight of w1 and w2
    # 这里可以直接调用pytorch的自动计算梯度（前提是在前面我们已经对w1 和w2 声明了requires_grad = True）
    # 需要注意的是，在pytorch中，所有Tensor的计算都是一张计算图（computation graph）
    # 会占用一定的内存，所以在更新下面的w1 w2 参数时，需要声明不再申请空间来存储w1 w2 的 grad
    with torch.no_grad():
        # 模型的所有参数都储存在param中，要进行参数更新时就直接批量更新，不用单独更新
        for param in model.parameters():
            param -= learning_rate * param.grad