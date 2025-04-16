import numpy as np
import math, torch
# import matplotlib.pyplot as plt


def integer_tent_map(length, keyx, keyp):                                                  # 1-D dynamic integer Tent function
    x = np.zeros((1, length));
    x[:, 0] = keyx                                                                         # set initial sates
    alpha = 2.0                                                                            # choose values for control parameters
    for i in range(length-1):                                                              # iterate the function
        ki = np.mod(keyp*(i+1), np.power(2.0, 32))
        gi = np.mod(x[:, i]+ki, np.power(2.0, 8))
        if gi >= 0.0 and gi < np.power(2.0, 7):
            x[:, i+1] = (-1)**i * alpha*gi+1
        elif gi >= np.power(2.0, 7) and gi <= np.power(2.0, 8)-1.0:
            x[:, i+1] =  alpha*(np.power(2.0, 8)-1-gi)

    return x[:, 99:-1]


def create_encryption_matrix(vector):                                                       # create a circulant matrix
    # n = 512
    # encryption_matrix = vector.reshape(n, n)

    n = vector.shape[0]                                                                    # 获取向量的长度
    encryption_matrix = torch.zeros((n, n))                                                # 创建一个全零的矩阵，大小为 (n, n)

    for i in range(n):                                                                     # 对于循环矩阵，每一行都是前一行的循环移位
        encryption_matrix[i, :] = torch.roll(vector, i, dims=0)

    return encryption_matrix


def encryption(input_tensor, enc_mat):                                                     # encryption function
    input_tensor = torch.matmul(input_tensor, enc_mat)

    return input_tensor


def decryption(input_tensor, enc_mat):                                                     # decryption function
    input_tensor = torch.matmul(input_tensor, torch.inverse(enc_mat))

    return input_tensor


# if __name__ == '__main__':
#     key_1 = 1.0; key_2 = 3.0; length = 110
#     colors = ['red', 'green', 'blue', 'yellow']
#
#     x1 = integer_tent_map(length, key_1)
#     print("\n密码流为:\n", x1)
#     cm = create_circulant_matrix(x1[0, :])
#     print("\n构建的循环矩阵为:\n", cm)
#     input_tensor = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]])
#     print("原始输入张量:\n", input_tensor)
#     encrypted_tensor = encryption(input_tensor, cm)
#     print("\n加密后的数据:\n", encrypted_tensor)
#     decrypted_tensor = decryption(encrypted_tensor, cm)
#     print("\n解密后的数据:\n", decrypted_tensor)
#
#     x2 = integer_tent_map(length, key_2)
#     print("\n密钥更新后密码流的差值:\n", x1 - x2)
#     inv_cm = np.linalg.inv(cm)
#     print("\n循环矩阵的逆矩阵为:\n", inv_cm)
#     res = np.round(inv_cm @ cm)
#     print("\n验证循环矩阵是否可逆:\n", res)
#     plt.scatter(range(length-100), x1[0, :], c=colors[0])
#     plt.scatter(range(length-100), x2[0, :], c=colors[2])
#     plt.title('1-D integer Tent map')
#     plt.xlabel('Iteration')
#     plt.ylabel('Value of x')
#     plt.show()