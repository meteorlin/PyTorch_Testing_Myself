import numpy as np
import torch
from FizzBuzz_Origin import fizz_buzz_encode as encode
from FizzBuzz_Origin import fizz_buzz_decode as decode

# 定义binary_encode()函数，将每一个数字转化为二进制数字。
# 因为后面定义的神经网络的输入有很多个输入神经元，如果单独输入一个数字训练效果就没有将每一个输入神经元都利用起来得好。
# num_digits是将数字由十进制转化为二进制的位数
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])

# 数字转化后的位数
NUM_DIGITS = 10

# 生成训练数据（101 ~ 2 ** NUM_DIGITS）
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([encode(i) for i in range(101, 2 ** NUM_DIGITS)]) # 注意这里表示类别的trY需要使用LongTensor，不能使用默认的Tensor（也就是Float）

# 定义隐藏层的维数
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4) # 返回四个类别的logits。
    # logits: 一个事件发生与该事件不发生的比值的对数，也就是：未归一化的概率
    # logits一般是“全连接层的输出，softmax的输入”
)

# 判断机器上的GPU是否可用
if torch.cuda.is_available():
    model = model.cuda()    # 如果机器有GPU可以在model后面接CUDA表示在GPU上训练

# 定义损失函数
# 对于分类问题（本问题为四分类问题），一般使用交叉熵作为损失函数
# 交叉熵：表示两种分布的相似度有多高
loss_fn = torch.nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 定义训练的批
BATCH_SIZE = 128
for epoch in range(2000):
    # 每间隔一个BATCH_SIZE抽取一次训练数据
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        # 如果GPU可用，则将训练数据转移到GPU上
        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        print ("Epoch", epoch, loss.item())

        # 优化
        # 1. 清空optimizeer的grad
        optimizer.zero_grad()
        # 2. 反向传播
        loss.backward()
        # 3. 梯度下降
        optimizer.step()

# 测试。测试的数据是1~101
testX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(1, 101)])
if torch.cuda.is_available():
    testX = testX.cuda()
with torch.no_grad():
    testY = model(testX)

# testY ：每一行表示每一个数字属于某一个类别的logits，数字越大代表可能性越大
# testY.max(1)：表示从testY这个tensor中的第一个维度取出每一行最大值，这样会返回两个维度，第一个[0]是logits，第二个[1]是最大logits在每一行所在的位置（用于后面的decode）
predictions = zip(range(1,101), testY.max(1)[1].cpu().data.tolist()) # 要转回到cpu上，再将其data取出后用list形式赋给predictions
print([decode(i, x) for i, x in predictions])
