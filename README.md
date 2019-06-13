# Disaster_Response_Pipeline  
## Introduction  
The project is to create a web interface where you have to enter a disaster message and it will categorize the type of message.
This will help in effective response and action to be taken to each message. This project is part of Data
Science Nanodegree Program by Udacity in collaboration with Figure Eight.The dataset is a combination of 
pre-labelled tweet and messages from real-life disaster.
</br>

<span>
<img src="https://github.com/rajatsharma369007/Disaster_Response_Pipeline/blob/master/image/image_1.png" width=400px height="280px" />
<img src="https://github.com/rajatsharma369007/Disaster_Response_Pipeline/blob/master/image/image_2.png" width=400px height="280px" />
</span>

## Getting Started
### Dependencies
* Python 3.5
* Scikit-Learn 0.21.2
* Pandas, Numpy
* NLTK, re
* Joblib
* SQLalchemy

### Executing Program:
* Run the following commands in the project's root directory to set up your database and model.  
  * To run ETL pipeline that cleans data and stores in database  
    <code> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db</code>

  * To run ML pipeline that trains classifier and saves.  
    <code>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</code>
* Run the following command in the app's directory to run your web app.  
  <code>python run.py</code>

* Go to http://localhost:3001/

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Issue/Bug
Please open issues on github to report bugs or make feature requests.

