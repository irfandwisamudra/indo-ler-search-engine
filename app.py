import json
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

with open('bert_embeddings.pkl', 'rb') as f:
    df = pickle.load(f)

with open('bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)

with open('data/all.meta.json') as f:
    meta_data = [json.loads(line) for line in f]

meta_df = pd.DataFrame(meta_data)
df = pd.merge(df, meta_df, on='id')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.form['query']
        count = request.form.get('count', '10')  # Jika tidak ada jumlah yang diminta, maka defaultnya adalah 10
        tokenized_query = user_query.split()
        scores = bm25.get_scores(tokenized_query)

        if count == 'All':
            count = len(scores)
        else:
            count = int(count)

        # Mengurutkan skor dari tertinggi ke terendah
        valid_scores_idxs = [idx for idx in scores.argsort()[::-1] if scores[idx] > 0]

        # Mengambil dokumen dengan skor tertinggi
        best_idxs = valid_scores_idxs[:count]
        results = []

        for idx in best_idxs:
            best_doc = df.iloc[idx]
            result = {
                'id': best_doc['id'],
                'text': best_doc['text'],
                'verdict': best_doc.get('verdict', 'N/A'),
                'indictment': best_doc.get('indictment', 'N/A'),
                'lawyer': best_doc.get('lawyer', 'N/A'),
                'owner': best_doc.get('owner', 'N/A'),
                'score': scores[idx]  # BM25 score
            }
            results.append(result)

        return render_template('results.html', results=results)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Error occurred while fetching results.")

@app.route('/details/<doc_id>')
def details(doc_id):
    try:
        doc_id = int(doc_id)
        doc = df[df['id'] == doc_id].iloc[0]
        
        # Pastikan ada hasil yang ditemukan
        if not doc.empty:
            result = {
                'id': doc['id'],
                'text': doc['text'],
                'verdict': doc.get('verdict', 'N/A'),
                'indictment': doc.get('indictment', 'N/A'),
                'lawyer': doc.get('lawyer', 'N/A'),
                'owner': doc.get('owner', 'N/A')
            }
            return render_template('details.html', result=result)
        else:
            return render_template('index.html', error="Document not found.")
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Error occurred while fetching details.")


if __name__ == '__main__':
    app.run(debug=True, port=5005)
