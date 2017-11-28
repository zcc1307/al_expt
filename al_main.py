import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class data_generator_model:
    pass

class al_model:
    pass

class data_model:
    pass

def data_generator(dgm):

    data = data_model()
    n_train = dgm.n_train
    n_test = dgm.n_test
    n_all = dgm.n_train + dgm.n_test
    d = dgm.d
    all_data = np.zeros((n_all, d+1))

    if dgm.data_source == "synthetic":
        if dgm.data_name == "twonorm":

            all_data[:,:d] = np.random.randn(n_all, d)

            for i in range(n_all):
                if np.random.rand(1) < 0.5:
                    all_data[i, 0] += 1
                    all_data[i, d] = -1
                else:
                    all_data[i, 0] += -1
                    all_data[i, d] = +1

        if dgm.data_name == "halfspace":
            w_star = np.random.randn(d)
            all_data[:, :d] = np.random.randn(n_all, d)
            for i in range(n_all):
                if np.dot(w_star, all_data[i,:d]) > 0:
                    all_data[i, d] = +1
                else:
                    all_data[i, d] = -1

        if dgm.data_name == "threshold":
            dgm.d = 1
            d = dgm.d
            all_data = np.zeros((n_all, 2))
            t_star = np.random.rand(1)*0.2 + 0.4
            all_data[:,0] = np.random.rand(n_all)
            for i in range(n_all):
                if all_data[i,0] <  t_star:
                    all_data[i,1] = -1
                else:
                    all_data[i,1] = +1

    data.x_train = all_data[:n_train, :d]
    data.x_test = all_data[n_train:n_train + n_test, :d]
    data.y_train = all_data[:n_train, d]
    data.y_test = all_data[n_train:n_train + n_test, d]
    data.n_train = n_train
    data.n_test = n_test
    data.d = d

    return data

def err_eval(model, x, y):
    if model == None:
        return 1

    y_pred = model.predict(x)
    accu = accuracy_score(y, y_pred)
    return 1 - accu

def active_learning(data, alm):
    queried = np.array([False for i in range(data.n_train)])
    model = None
    n_phases = data.n_train/alm.batch_size;
    train_err = np.zeros(n_phases)
    test_err = np.zeros(n_phases)

    for i in range(n_phases):
        selected = selective_sample(alm, model, data, queried)

        for sel in selected:
            queried[sel] = True

        if len(set(data.y_train[queried])) > 1:
            model = svm.SVC()
            model.fit(data.x_train[queried, :], data.y_train[queried])

        train_err[i] = err_eval(model, data.x_train, data.y_train)
        test_err[i] = err_eval(model, data.x_test, data.y_test)

    return train_err, test_err

def selective_sample(alm, model, data, queried):
    selected = []

    if alm.query_method == "random" or model == None:
        pred = np.random.rand(data.n_train)
    elif alm.query_method == "uncertainty" and model != None:
        pred = model.decision_function(data.x_train)

    abs_pred = abs(pred)
    sort_idx = np.argsort(abs_pred)
    count = 0;

    for i in range(data.n_train):
        if count == alm.batch_size:
            break

        if queried[sort_idx[i]] == False:
            selected.append(sort_idx[i])
            count += 1

    return selected



if __name__== "__main__":

    dgm = data_generator_model()
    dgm.n_train = 100
    dgm.n_test = 2000
    dgm.d = 2
    dgm.data_source = "synthetic"
    #dgm.data_name = "twonorm"
    #dgm.data_name = "halfspace"
    dgm.data_name = "threshold"

    alm = al_model()
    alm.query_method = "uncertainty"
    alm.batch_size = 1

    data = data_generator(dgm)

    print data.n_train
    training_curve, test_curve = active_learning(data, alm)


    alm.query_method = "random"
    training_curve_r, test_curve_r = active_learning(data, alm)

    print training_curve
    print training_curve_r

    fig = plt.figure()
    plt.hold(True)
    #plt.plot(test_curve, 'b')
    #plt.plot(test_curve_r, 'r')
    active_plot, = plt.plot(test_curve, 'b', label="active")
    passive_plot, = plt.plot(test_curve_r, 'r', label="passive")
    plt.legend(handles=[active_plot, passive_plot])
    plt.show()
