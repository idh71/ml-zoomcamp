The Problem

My problem was to build a classifier that when trained on a dataset of 26,000 individiduals with 16 variables (age, workclass, education, race,
marital status, occupation, etc.) can successuly predict based on these variables whether a given idiviual makes more or less than $50,0000 per year.


Instructions for running the project

1. make sure you have bentoml installed.  Run the train.py file in the same directory where the adult.data is located.  This will prepare the data, 
 train the model and save it.

2. Run the following command in the same folder:

    bentoml build
  
  this will build the build the bento with the model you just trained and saved. Once the bento has been built you will see this message(with a new tag):
    
    Successfully built Bento(tag="over_50k_classifier:ji6lpydayou3nct7")
    
  copy the tag and use it to run:
  
    bentoml containerize over_50k_classifier:ji6lpydayou3nct7
    
  now to tun the model in a docker container:
  
    docker run -it --rm -p 3000:3000 over_50k_classifier:ji6lpydayou3nct7
    
 3. Open a browser to localhost:3000 to test the service with the swagger api.  click the drop down on the far right of /classify under
    Service APIs, then click the button "Try it out".  You can use this data to test the service:
    
    ``{"age": 25,
 "workclass": "private",
 "fnlwgt": 102476,
 "education": "10th",
 "education_num": 13,
 "marital_status": "never_married",
 "occupation": "farming_fishing",
 "relationship": "own_child",
 "race": "white",
 "sex": "male",
 "capital_gain": 27828,
 "capital_loss": 0,
 "hours_per_week": 50,
 "native_country": "united_states",
 "us_native": "united_states"}``
 
   once you click execute you can scroll down to see the response body.  it should look something like this:
  
  ``{
  "status": "UNDER 50K"
}``

  if you change the value of capital_gain to 0 and run it again you will get a different response.  You can play with the values of the
  variables to see how different values affect the prediction. I hve included screen shots of myself using the service to make predictions.


  
    
  
    
  
  


