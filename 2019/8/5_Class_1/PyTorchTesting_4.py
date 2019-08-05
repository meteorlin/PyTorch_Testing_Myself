import torch.nn as nn

# 超变量定义（hypeparameters）
# 输入个数 N = 64
# 输入维数 D_in = 1000
# 中间隐藏层维数 H = 100
# 输出维数 D_out = 10
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

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

'''
    定义一个优化函数，使得pytorch根据我们选择的方法，自动协助我们对参数进行update
'''
# 对于任何一个优化器，都需要把模型的参数（model.parameters()）和学习率（learning_rate）传递给它

# 可以使用normal_对model里面的连接权重和正态分布进行拟合，使得网络的参数分布符合正态分布，这样训练时可能会较快收敛
# 根据上面的model的定义，第一、三层是线性神经元层，第二层是激活函数ReLU()，所以应该对第一、三层的参数进行正则化。
# 定义学习率（learning_rate）
# learning_rate = 1e-6
# torch.nn.init.normal_(model[0].weight)
# torch.nn.init.normal_(model[2].weight)
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 针对Adam优化方法而言，1e-3到1e-4是比较好的学习率
# 对于Adam而言，可能不使用torch.nn.init.normal_训练效果会好一点，而SGD就需要进行normal_
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# 可以直接在进入神经网络训练之前定义好loss公式   均方误差L2 (MSELoss)
loss_fn = nn.MSEloss(reduction='sum')

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
    optimizer.zero_grad()
    # compute the gradient
    # 使用backward()自动计算loss中相关参数的梯度
    loss.backward()

    # update model parameters
    optimizer.step()