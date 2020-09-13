from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():

    
    return render_template("heatmaps.html")


@app.route("/ml", methods = ["POST"])
def ml():
    message = ""
    msg11 = request.form.get("name1")
    msg1 = int(msg11)
    msg22 = request.form.get("name2")
    msg2 = int(msg22)
    msg33 = request.form.get("name3")
    msg3 = int(msg33)
    msg44 = request.form.get("name4")
    msg4 = int(msg44)
    msg55 = request.form.get("name5")
    msg5 =  int(msg55)

    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import linear_model
    from sklearn.utils import shuffle

    data = pd.read_csv("student-mat.csv", sep=";")

    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    predict = "G3"

    X = np.array(data.drop([predict], 1))
    y = np.array(data[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)

    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])

    ## Predicting a value
    #x = int(input("Enter some factors"))
    prediction1 = linear.predict([[msg1, msg2, msg3, msg4, msg5]])
    if prediction1 > 15:
        message = "The machine learning model predicts you are at great risk in your current area. Please check the how to stay safe page to learn safety precautions."
    else:
        message = "The machine learning model predicts you area is at low risk of air pollution. However in case, it is recommended to check the safety precautions guide."
    print(prediction1)
    print(message)
    return render_template("safetyprecautions.html", message = message)




if __name__ == "__main__":
    app.run(port=5000, debug=True)
