import cmath
# 一维快速傅里叶变换（递归实现）
def fft_1d(signal):
    N = len(signal)
    if N <= 1:
        return signal
    even = fft_1d(signal[0::2])
    odd = fft_1d(signal[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# 一维逆傅里叶变换
def ifft_1d(signal):
    N = len(signal)
    conjugated = [x.conjugate() for x in signal]
    transformed = fft_1d(conjugated)
    return [x.conjugate() / N for x in transformed]

# 二维快速傅里叶变换
def fft_2d(matrix):
    M, N = len(matrix), len(matrix[0])
    # 对每一行进行傅里叶变换
    fft_rows = [fft_1d(row) for row in matrix]
    # 转置后对每一列进行傅里叶变换
    fft_columns = [fft_1d(col) for col in zip(*fft_rows)]
    # 再次转置回原格式
    return [list(row) for row in zip(*fft_columns)]

# 二维逆傅里叶变换
def ifft_2d(matrix):
    M, N = len(matrix), len(matrix[0])
    # 对每一行进行逆傅里叶变换
    ifft_rows = [ifft_1d(row) for row in matrix]
    # 转置后对每一列进行逆傅里叶变换
    ifft_columns = [ifft_1d(col) for col in zip(*ifft_rows)]
    # 再次转置回原格式
    return [list(row) for row in zip(*ifft_columns)]