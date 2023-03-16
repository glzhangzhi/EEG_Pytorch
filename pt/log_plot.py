import matplotlib.pyplot as plt

with open('training_1.log', 'r') as f:
    log = f.readlines()


x, y = [], []
for i in log:
    if 'epoch' not in i:
        continue
    epoch, _, h = i.split('epoch ')[-1].split(' ')
    x.append(int(epoch))
    y.append(float(h))

plt.plot(x, y, '-')
plt.show()