# 定义fizz_buzz_encode(i)函数，输入一个数字i，判断是应该输出fizz,buzz,fizzbuzz还是原数
def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    if i % 5 == 0: return 2
    if i % 3 == 0: return 1
    else: return 0

# 定义fizz_buzz_decode(i, prediction)函数，i表示输入的数字，prediction表示该数字属于第几类数字（该数字属于的类别编号由fizz_buzz_encode(i)函数给出）
def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def helper(i):
    print(fizz_buzz_decode(i,fizz_buzz_encode(i)))

for i in range(1,100):
    helper(i)