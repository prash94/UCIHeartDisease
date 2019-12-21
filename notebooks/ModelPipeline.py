from HeartDiseaseProject.src.ProjectConfig.Config import LogisticRegression, accuracy_score
import pickle
import pathlib


def lr_model(dt, tgt, plty, clswt):
    clf = LogisticRegression(penalty=plty, class_weight=clswt)
    clf.fit(dt, tgt)
    model_accuracy = accuracy_score(tgt, clf.predict(dt))
    print(model_accuracy)

    filename = 'model1.pickle'
    pickle.dump(clf, open(filename, 'wb'), protocol=2)
    model = pickle.load(open(filename,'rb'))
    print(model.predict([[22584, 2, 178, 95, 130, 90, 3, 3, 0, 0, 1]]))


