import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("140train.csv")
df.columns = ["target", "id", "date", "flag", "user", "text"]
df=df.drop(["id", "date", "flag", "user"], axis=1)

df['target']=df['target'].map({0:0, 4:1})
print(df.head())

train, test = train_test_split(df, test_size=0.3, random_state=42)
print(test.head())

os.mkdir("140")
os.mkdir("140/train")
os.mkdir("140/test")

os.mkdir("140/train/0")
os.mkdir("140/train/1")
os.mkdir("140/test/0")
os.mkdir("140/test/1")

i=0
for t in test.itertuples():
    f = open("140/test/"+str(t[1])+"/"+str(i)+".txt", encoding="utf8", mode="w")
    f.write(t[2])
    f.close()
    i+=1
for t in train.itertuples():
    f = open("140/train/"+str(t[1])+"/"+str(i)+".txt", encoding="utf8", mode="w")
    f.write(t[2])
    f.close()
    i+=1
