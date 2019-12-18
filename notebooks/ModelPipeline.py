from HeartDiseaseProject.src.ProjectConfig.Config import LogisticRegression, accuracy_score



def LRBaseline(dt,tgt,plty,clswt):
    clf = LogisticRegression(penalty=plty, class_weight=clswt)
    clf.fit(dt, tgt)
    Model_Accuracy = accuracy_score(tgt, clf.predict(dt))
    print(Model_Accuracy)