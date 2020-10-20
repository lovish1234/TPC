import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()

# trained folder details
# parser.add_argument('--prefix', default='tmp', type=str)
# parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument(
    '--file', default='certain_L2_train.csv', nargs='*', type=str)
parser.add_argument('--attribute', default='local/loss', type=str)


if __name__ == '__main__':

    args = parser.parse_args()
    #print (str(args.attribute))
    legList = []
    ax = plt.gca()

    for i in range(len(args.file)):

        df = pd.read_csv(args.file[i])
        df = df.loc[df['metric'] == args.attribute]

        df.plot(kind='line', x='step', y='value', ax=ax)
        plt.xlabel('Iteration')
        plt.ylabel('Top 5 Accuracy')

        legList.append(args.file[i].split('/')[-1].split('.')[0])

    ax.legend(legList)
    plt.savefig('output.png')
    #df.plot(kind='line',x='name',y='num_pets', color='red', ax=ax)
