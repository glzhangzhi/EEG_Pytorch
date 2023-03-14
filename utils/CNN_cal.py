# # input size
H1, W1, D1 = (174, 24, 16)

# # parameter
K = 32
F = 5
S = 1
P = 0

W2 = int((W1 - F + 2 * P) / S + 1)
H2 = int((H1 - F + 2 * P) / S + 1)  
D2 = K

P_same = ((W1 - 1) * 2 - W1 + F) / 2, ((H1 - 1) * 2 - H1 + F) / 2

print(f'input size: ({H1}, {W1}, {D1}), features: {H1 * W1 * D1}')
print(f'output size: ({H2}, {W2}, {D2}), features: {H2 * W2 * D2}')

# print('卷积减半参数')
# for f in range(2, 10):
#     for s in range(1, 10):
#         for p in range(0, 1000):
#             if int((H1 - f + 2 * p) / s + 1) == H1/2 and int((W1 - f + 2 * p) / s + 1) == W1/2:
#                 print(f'{f=}, {p=}, {s=}')

# print('卷积不变参数')
# for f in range(2, 10):
#     for s in range(1, 10):
#         for p in range(0, 1000):
#             if int((H1 - f + 2 * p) / s + 1) == H1 and int((W1 - f + 2 * p) / s + 1) == W1:
#                 print(f'{f=}, {p=}, {s=}')

# print('反卷积参数翻倍')
# for f in range(2, 10):
#     for s in range(1, 10):
#         for p in range(0, 1000):
#             if (H1 - 1) * s - 2 * p + f == H1 * 2 and (W1 - 1) * s - 2 * p + f == W1 * 2:
#                 print(f'{f=}, {p=}, {s=}')

# print('反卷积不变参数')
# for f in range(2, 10):
#     for s in range(1, 10):
#         for p in range(0, 1000):
#             if (H1 - 1) * s - 2 * p + f == H1 and (W1 - 1) * s - 2 * p + f == W1:
#                 print(f'{f=}, {p=}, {s=}')
