# AnimeRecommenderSystem
A anime recommender system that recommends the top animes. The search option enables user to search for the favourite anime and this gives the information and anime recommendations related to the anime. 


## Demo web application

![animerecomgif](https://user-images.githubusercontent.com/64649488/208632135-e5d5293b-d104-4b7b-a0e7-098f90ab311c.gif)

## Installation 
clone the repository
https://github.com/Shouvik-7/AnimeRecommenderSystem

Go to the following website and download the files to the model folder of the cloned repository.
https://huggingface.co/siebert/sentiment-roberta-large-english/tree/main

## Running the application

install the requirements.txt file

pip install -r requirements.txt

run the app.py file

python app.py

## Features
A content based recommender system is used. This uses features tags consists of words from synopsis, genre and rating which are vectorized to form the feature set. Cosine similarity is used to find similar animes. 

Jikan API is used to collect the reviews of the searched anime and sentiment analysis is run on the reviews. Pretrained sentiment-roberta-large-english is used to get the positive or negative labels from the reviews. 
