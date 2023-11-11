import multiprocessing
import time

# 全局共享变量
shared_variable = 0

def func_a():
    global shared_variable
    for _ in range(5):
        shared_variable += 1
        time.sleep(1)
        print("Func A: Shared Variable =", shared_variable)

def func_b():
    global shared_variable
    for _ in range(5):
        shared_variable -= 1
        time.sleep(1)
        print("Func B: Shared Variable =", shared_variable)

if __name__ == "__main__":
    # 创建两个进程
    process_a = multiprocessing.Process(target=func_a)
    process_b = multiprocessing.Process(target=func_b)

    # 启动进程
    process_a.start()
    process_b.start()

    # 等待两个进程执行完成
    process_a.join()
    process_b.join()

    print("Main Process: Shared Variable =", shared_variable)
