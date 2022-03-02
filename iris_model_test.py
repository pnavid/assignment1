import iris_model


def test(sepal_length, sepal_width, petal_length, petal_width):

    y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
    trained_model = iris_model.training_model()
    prediction_value = trained_model.predict(y_pred)
    return prediction_value
