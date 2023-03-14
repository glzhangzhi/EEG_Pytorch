import matplotlib.pyplot as plt

with open('output_data/training.log', 'r') as f:
    log = f.readlines()

x, y = [], []
cnt = 1
for i in log:
    i = i.split(', ')[2].split(': ')[-1]
    x.append(int(cnt))
    y.append(float(i))
    cnt += 1

plt.plot(x, y, '.')
plt.show()