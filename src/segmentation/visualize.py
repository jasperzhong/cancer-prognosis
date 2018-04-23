import re
import matplotlib.pyplot as plt



def plot_loss():
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
        plt.ylabel('NLLoss2d')
        plt.legend(handles=[l1,l2], labels=['train', 'validation'], loc='best' )
        plt.show()

def plot_accu():
    with open("/home/yuchen/Programs/cancer-prognosis/accu.txt") as file:
        contents = file.read()
        pattern = re.compile("\d+.\d+")
        contents = pattern.findall(contents)

        accu = []
        for content in contents:
            accu.append(float(content))
        length = len(accu)
        train =  [accu[i] for i in range(0,length, 2)]
        test =  [accu[i] for i in range(1,length, 2)]
        l1, = plt.plot(range(length//2), train, 'r')
        l2, = plt.plot(range(length//2), test, 'b')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim((0.4,1))
        plt.legend(handles=[l1,l2], labels=['train', 'validation'], loc='best' )
        plt.show()
        
plot_accu()