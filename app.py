from flask import Flask, render_template, request
import pickle
import numpy as np
import sentimentReviews

popularanimes = pickle.load(open('popularanimes.pkl','rb'))
anime_df = pickle.load(open('anime_df.pkl','rb'))
similarityMatrix = pickle.load(open('similarityMatrix.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           anime_title = list(popularanimes['title'].values),
                           rating = list(popularanimes['score'].values),
                           img_url =list(popularanimes['img_url'].values)
                           )

@app.route('/recommend')
def recommendUI():
    return render_template('recommender.html')

@app.route('/recommend_anime', methods=['post'])
def recommend():
    user_input = str(request.form.get('user_input'))
    try:
        givenAnime = anime_df[anime_df['title'] == user_input]
        animeData = []
        animeData.extend(list(givenAnime['title'].values))
        animeData.extend(list(givenAnime['score']))
        animeData.extend(list(givenAnime['img_url']))
        animeData.extend(list(givenAnime['synopsis']))
        animeData.extend(list(givenAnime['genre']))

        givenAnimeIndex = givenAnime.index[0]
        similarity = similarityMatrix[givenAnimeIndex]
        animeList = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[1:6]
        data = []
        for anime in animeList:
            item = []
            tempdf = anime_df.iloc[anime[0]]
            item.append(tempdf['title'])
            item.append(tempdf['score'])
            item.append(tempdf['img_url'])
            data.append(item)
        print(data)

        uid = list(givenAnime['uid'].values)[0]
        reviewdf = sentimentReviews.getReviewsSentiments(uid)
        reviews = []
        for j in range(3):
            reviews.append([reviewdf.iloc[j]['text'],reviewdf.iloc[j]['label'],reviewdf.iloc[j]['pred']])
    except:
        data = []
        animeData = []
        reviews = []
    return render_template('recommender.html', data=data, animeData=animeData, reviews=reviews)


if __name__ == '__main__':
    app.run(debug=True)