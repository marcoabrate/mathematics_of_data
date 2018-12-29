import matplotlib.pyplot as plt
import pickle
import os

def plot():
    outdir='lr05'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    os.system('python main.py --optimizer sgd --learning_rate 0.5 --output='+outdir+'/sgd.pkl')
    os.system('python main.py --optimizer momentumsgd --learning_rate 0.5 --output='+outdir+'/momentumsgd.pkl')
    os.system('python main.py --optimizer rmsprop --learning_rate 0.5 --output='+outdir+'/rmsprop.pkl')
    os.system('python main.py --optimizer adam --learning_rate 0.5 --output='+outdir+'/adam.pkl')
    os.system('python main.py --optimizer adagrad --learning_rate 0.5 --output='+outdir+'/adagrad.pkl')
    
    optimizers = ['sgd', 'momentumsgd', 'rmsprop', 'adam', 'adagrad']
    
    # Plots the training losses.
    for optimizer in optimizers:
       data = pickle.load(open(outdir+'/'+optimizer+".pkl", "rb"))
       plt.plot(data['train_loss'], label=optimizer)
    plt.ylabel('Trainig loss')
    plt.xlabel('Epochs')
    plt.ylim(0, 4)
    plt.legend()
    plt.savefig(outdir+'/loss.pdf')
    plt.show()
    
    # Plots the training accuracies.
    for optimizer in optimizers:
        data = pickle.load(open(outdir+'/'+optimizer+".pkl", "rb"))
        plt.plot(data['train_accuracy'], label=optimizer)
    plt.ylabel('Trainig accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(outdir+'/accuracy.pdf')
    plt.show()

if __name__=="__main__":
    plot()