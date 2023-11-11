import numpy as np

# 生成规模为(64, 3489)的矩阵，每个元素在(-0.01, 0.03)范围内，并保留两位小数
matrix = np.round(np.random.uniform(-0.01, 0.03, size=(64, 64)), 2)

# 保存矩阵到文件，每一行保存矩阵的一行，并使用空格进行元素分隔
np.savetxt('lsf_w.txt', matrix, fmt='%.2f', delimiter=' ', newline='\n')

print("Matrix saved to lsf_W.txt")
