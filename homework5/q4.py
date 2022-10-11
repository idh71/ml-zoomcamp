import requests



url = 'http://localhost:9696/predict'
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
requests.post(url, json=client).json()




url = 'http://localhost:9696/predict'

# customer_id = 'xyz-123'



response = requests.post(url, json=client).json()
print(response)

# if response['churn'] == True:
#     print('sending promo email to %s' % customer_id)
# else:
#     print('not sending promo email to %s' % customer_id)