from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import imageio
from nltk.corpus import stopwords
from mega import Mega
from os.path import exists
from textblob import TextBlob
import re
import string

#dataset download
if(not exists("140train.csv")):
    mega = Mega()
    m = mega.login()
    try:
        m.download_url('https://mega.nz/file/hGoyzZRS#wcIdpR7mOfPXslMEbkvOxLrSpkTL3qZSq-8xyy3nbBA')
    except:
        pass

#load and prepare the dataset
df = pd.read_csv("140train.csv", encoding='latin-1')
df.columns = ["target", "id", "date", "flag", "user", "text"]
df=df.drop(["id", "date", "flag", "user"], axis=1)
df['target']=df['target'].map({0:0, 4:1}) 

def cleanText(text):
  text = re.sub(r'@[A-Za-z0-9]+', '', text)
  text = re.sub(r'#', '', text)
  text = re.sub(r'https?:\/\/\S+', '', text)
  text = re.sub(r'&quot', '', text)
  text = re.sub(r'&amp', '', text)
  text = re.sub(r'&lt', '', text)
  text = re.sub(r'&gt', '', text)
  return text

df['text'] = df['text'].apply(cleanText)

#remove stop words (not relevant for wordcloud...)
nltk.download('stopwords')
stops = stopwords.words('english')
wholeText = ' '.join([t for t in df['text']])
wholeTextNoStops = ' '.join([word for word in wholeText.split() if word not in stops])

#display wordcloud
brain_mask = imageio.imread('brainmask.jpg')
wordcloud = WordCloud(width=700, height=700, colormap='prism', background_color='black', mask=brain_mask)
wordcloud = wordcloud.generate(wholeTextNoStops)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Wordcloud")
plt.show()


#labels distribution
histogram = df['target'].value_counts()
plt.axis('off')
labs = ['Positive', 'Negative']
values = list(histogram.to_dict().values())
plt.title("Labels distritution")
plt.pie(values, labels = labs,autopct='%1.2f%%')
plt.show()
