import re
import matplotlib.pyplot as plt

with open('/home/yuchen/Programs/cancer-prognosis/train.log', 'r') as file:
    contents = file.read()

pattern = re.compile("-?\d+\.\d+")
results = pattern.findall(contents)

length = len(results)
for i in range(length):
    results[i] = float(results[i])

train =  [results[i] for i in range(0,length, 2)]
test =  [results[i] for i in range(1,length, 2)]


l1, = plt.plot(range(length//2), train, 'r')
l2, = plt.plot(range(length//2), test, 'b')
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
plt.legend(handles=[l1,l2], labels=['train', 'test'], loc='best' )
plt.show()
