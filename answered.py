import json
import operator
import string
import sys

# import ipdb
from sklearn import tree


def extract_features(data, featureKeys=None):
	samples = []

	for d in data:
		w1 = [w.strip().lower() for w in d["question_text"].split()]
		w2 = [t["name"].strip().lower() for t in d["topics"]]
		w3 = ["".join([ch for ch in t["name"].strip().lower() if ch in string.ascii_lowercase]) for t in d["topics"]]
		sample = [
			d["anonymous"],
			d["context_topic"]["followers"] if hasattr(d["context_topic"], "followers") else 0,
			sum(t["followers"] for t in d["topics"]),
			len(set(w1) & set(w2) & set(w3)),
			# len(set(w1) & set(w2) & set(w3)) > 0,
			# len(set(w1) & set(w2) & set(w3)) > 1,
			# len(set(w1) & set(w2) & set(w3)) > 2,
			# len(set(w1) & set(w2) & set(w3)) > 5,
			# len(set(w1) & set(w2) & set(w3)) > 10,
		]
		samples.append(sample)

	return samples


if __name__ == "__main__":
	N = 0

	lines = []
	lines = [l for l in sys.stdin]
	# with open("answered_data_10k.in", "r") as f:
	# 	lines = f.readlines()
	lines = [l.strip() for l in lines]

	N = int(lines[0])
	C = int(lines[N+1])

	data = [json.loads(l) for l in lines[1:N] if l]

	start, end = N + 1 + 1, N + 1 + 1 + C
	data2 = [json.loads(l) for l in lines[start:end]]

	trainingData = extract_features(data)
	trainingDataClassLabels = [d["__ans__"] for d in data]

	clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
	clf = clf.fit(trainingData, trainingDataClassLabels)

	queries = extract_features(data2)
	predictions = clf.predict(queries)

	for d in zip(data2, predictions):
		res = {
			"question_key": d[0]["question_key"],
			"__ans__": bool(d[1]),
		}
		print json.dumps(res)
