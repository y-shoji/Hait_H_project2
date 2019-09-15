from sklearn.neighbors import KNeighborsClassifier
import pickle


def output_color_Coordinate(input,filename):  
    with open(filename, mode='rb') as fp:
        x=[]
        x.append(input)
        model = pickle.load(fp)

    return model.predict(x)


def _main():
    x=[[220.0, 218.0, 213.0, 0, 0, 0]]
    filename = 'static/model/model.pickle'
    print(output_color_Coordinate(x,filename))

if __name__=='__main__':
    _main()