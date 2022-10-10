import pickle



model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)





def predict():
    customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return result

print(predict())



if __name__ == "__main__":
    print(predict())