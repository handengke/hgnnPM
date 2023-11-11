import threading

# 定义要执行的多个代码块
def code_block_1():
    global a
    # 第一个代码块的逻辑
    for idx in range(len(a)):
        if a[idx]==1:
            a[idx]-=1

def code_block_2():
    global a
    # 第二个代码块的逻辑
    for idx in range(len(a)):
        a[idx]+=1

a=[1,1,1,1]

# 创建多个线程，每个线程执行一个代码块
thread1 = threading.Thread(target=code_block_1)
thread2 = threading.Thread(target=code_block_2)

# 启动线程
thread1.start()
thread2.start()

# 等待线程执行完成
thread1.join()
thread2.join()

# # 获取执行结果
# result1 = thread1.result()
# result2 = thread2.result()