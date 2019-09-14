from sklearn.neighbors import KNeighborsClassifier
import pickle


def output_color_Coordinate(x,filename):  
    with open(filename, mode='rb') as fp:
        model = pickle.load(fp)

    return model.predict(x)




def _main():
    x=[[0,0,0,0,0,0]]
    filename = 'ML/model.pickle'
    print(output_color_Coordinate(x,filename))

if __name__=='__main__':
    _main()