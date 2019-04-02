import sys
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from bert_serving.client import BertClient
    from sklearn.metrics import pairwise_distances_argmin_min
    from flask import Flask,jsonify,request
    from flask_cors import CORS
    from nltk import sent_tokenize
except ImportError:
    sys.exit('Error importing modules')

app = Flask(__name__)
CORS(app)

bc = BertClient(check_length=False)

@app.route('/summary', methods=['POST'])
def summary():
    req = request.get_json(force=True)
    text = req['text']
    sent_list = sent_tokenize(text)
    sent_list = [sent for sent in sent_list if len(sent)>20]
    encoded = bc.encode(sent_list).tolist()

    n_clusters = int(np.ceil(len(encoded)**0.5))

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(encoded)

    print(n_clusters, len(encoded))
    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, encoded)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summ = ' '.join([sent_list[closest[idx]] for idx in ordering])
    output = {"summary":summ}
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)