# SentimentAnalysisSICD

Il progetto è stato realizzato per il corso di Sistemi Intelligenti per la Comunicazione Digitale AA 2020-2021 da Alessandro Paparella e Francesco Lobascio.

Il progetto ha come obiettivo finale quello di svolgere sentiment analysis su contenuti pubblicati sul web, in particolare sui post e commenti pubblicati su reddit.com di un dato topic o discussione. Il modello di classificazione usato è stato allenato sul modello di rete neurale Bert, nello specifico il modello pre-allenato "bert-base-uncased".

## Fasi del progetto
### Esplorazione dataset
In questa fase abbiamo esplorato il dataset scelto ovvero "sentiment140", un dataset con label binarie e 1.6 milioni di righe contenenti tweet su opinioni riguardanti svariati argomenti. Abbiamo scoperto che il dataset è perfettamente bilanciato e visualizzando la wordcloud abbiamo avuto conferma che il dataset fosse generico e le parole più frequenti mostravano una certa polarità tra cui "love", "bad", "hate", "nice"...

### Training e valutazioni
Questa fase è stata ripetuta diverse volte per cercare di ottenere delle performance migliori, in particolare abbiamo esteso man mano il dataset fino al totale delle entry presenti (arrivando a rendere necessario l'uso delle TPU di google colab). In tutti i casi è stato usato uno split 80/20 per training e validazione.

Per quanto riguarda il dataset ridotto utilizzando la GPU abbiamo usato fino a un massimo di 40000 righe per il training, 8000 per la validazione e 20000 per la valutazione e abbiamo notato una situazione di overfitting dato che abbiamo osservato i seguenti valori:

| Fase       | Loss   | Accuracy |
|------------|--------|----------|
| Training   | 0.1149 | 0.9582   |
| Validation | 0.6697 | 0.8292   |
| Evaluation | 0.7030 | 0.8180   |

Di conseguenza abbiamo deciso di estendere il dataset coprendo tutti gli esempi usando il seguente split training-testing 70/30 sul totale e a sua volta il training splittato con la validazione 80/20. In questo caso abbiamo ottenuto dei miglioramenti ed eliminato l'overfitting:

| Fase       | Loss   | Accuracy |
|------------|--------|----------|
| Training   | 0.3016 | 0.8721   |
| Validation | 0.3291 | 0.8598   |
| Evaluation | 0.3295 | 0.8594   |

Infine abbiamo anche testato il modello "bert-base-cased" dal momento che in genere gli utenti in base allo stato d'animo possono utilizzare caratteri maiuscoli o meno, tuttavia non abbiamo riscontrato miglioramenti e abbiamo deciso di continuare a utilizzare il modello "bert-base-uncased".


## 2. Struttura
```
|-- SentimentAnalysisReddit_GPU.py
|-- SentimentAnalysisReddit_TPU.py
|-- README.md
|-- dataExploration.py
|–– trainingGPU.py
|–– trainingTPU.py
|-- requirements.txt
```

Sono presenti due file differenti (SentimentAnalysisReddit_GPU.py eSentimentAnalysisReddit_TPU.py) per quanto riguarda lo scraper di reddit poiché il modello allenato sulle TPU prende in input un formato di dati differente e inoltre è necessario scaricare prima il modello pre-allenato e poi caricare i pesi ottenuti con il training, a differenza del modello allenato con GPU il quale è stato serializzato per intero.

## 3. Requisiti

