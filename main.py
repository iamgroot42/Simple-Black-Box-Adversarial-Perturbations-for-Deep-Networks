import keras
from keras.models import load_model
import sys
import numpy as np


def top_k_predicitons(model, X, k):
	proba = model.predict_proba(X)
	labels = np.argsort(x, axis=1)
	selected_labels = labels[:,k]
	return selected_labels


def randadv(model, img, label, p, U):
	# Expects image in th-ordering
	# Assumes img is a good image (as defined in the paper)
	critical = 0
	for _ in range(U):
		x = np.random.choice(img.shape[1])
		y = np.random.choice(img.shape[2])
		perturbed = np.copy(img)
		perturbed[:,x,y] = np.sign(perturbed[:,x,y]) * p
		if label not in top_k_predicitons(model, [img], 1):
			critical += 1
	return float(critical) / U


if __name__ == "__main__":
	model = load_model(sys.argv[1])
