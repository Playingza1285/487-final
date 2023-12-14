# EECS 487 Final Project - Bias Detecton Within Mass Media

Group Members: Andrew Shin, William Chen, Vincent Weng

## Problem Description

In an era where the media plays a pivotal role in shaping public opinion, identifying bias has become increasingly important. For much of the public, the media is the main, if not only, source of information and influences the public’s perception of most issues or ideas. As a result, our team has focused this project on recognizing the political standings of the authors and discerning the underlying biases in articles. 
If successful, this initiative aims to promote transparency in news outlets and provide insight to the public on the influence the media has on their opinions.

Our team investigates author bias by analyzing the language elements like word choice or syntax in the input text data and then predicting the author’s ideological positions as the main output. 
This can be a vital tool for detecting bias within political spheres and discerning biased opinions within articles overall.

## Dataset
For our purposes, our team utilized a dataset containing information on 37,554 articles from www.allsides.com. These articles are collected and converted to JSON formats from this public github repository: https://github.com/ramybaly/Article-Bias-Prediction.

Example JSON Format:
```
{
    "topic": "terrorism",
    "source": "New York Times - News",
    "bias": 0,
    "url": 
    "http://www.nytimes.com/”
    "title": "Bomb Suspect Changed After Trip Abroad, Friends Say",
    "date": "2016-09-20",
    "authors": "N. R. Kleinfield",
    "content_original": "Besides his most recent trip to      Quetta, Mr. Rahami visited Karachi, Pakistan, in 2005. Both of those cities\u2019 reputations have become entwined with the militant groups…”
    "source_url": "www.nytimes.com",
    "bias_text": "left",
    "ID": "004Gt3gcsotuiYmz"
}
```
