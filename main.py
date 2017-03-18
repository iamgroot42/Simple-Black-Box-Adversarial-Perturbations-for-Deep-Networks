import keras
from keras.models import load_model
import sys
import numpy as np


def score(model, imgs, c):
	return model.predict_proba(imgs)[:,c]


def cyclic(value, lb, ub):
	if value < lb:
		return value + (ub-lb)
	elif value > ub:
		return value - (ub-lb)
	else:
		return value


def top_k_predicitons(model, X, k):
	proba = model.predict_proba(X)
	labels = np.argsort(x, axis=1)
	selected_labels = labels[:,k]
	return selected_labels


def perturbed(img, x, y, p):
	perturbed = np.copy(img)
	perturbed[:,x,y] = np.sign(perturbed[:,x,y]) * p
	return perturbed


def randadv(model, img, label, p, U):
	# Expects image in th-ordering
	# Assumes img is a good image (as defined in the paper)
	critical = 0
	for _ in range(U):
		x = np.random.choice(img.shape[1])
		y = np.random.choice(img.shape[2])
		perturbed = perturbed(img, x, y, p)
		if label not in top_k_predicitons(model, [perturbed], 1):
			critical += 1
	return float(critical) / U


def locsearchadv(model, img, p, r, d, t, k, R, label):
	dim1, dim2 = image.shape[1], image.shape[2]
	PX, PY = np.random.choice(range(int(dim1)),int(dim1*0.1)), np.random.choice(range(int(dim2)),int(dim2*0.1))
	I = np.copy(img)
	i = 1
	while i <= R:
		# Computing the function g using the neighborhood
		L = []
		for j in range(len(PX)):
				L.append(perturbed(I, PX[j], PY[j], p))
		L = np.array(L)
		scores = -1 * score(model, L, label)
		sorted_L = np.argsort(scores)
		PX = (PX[sorted_L])[:t]
		PY = (PY[sorted_L])[:t]
		# Generation of the perturbed image I
		for j in range(len(PX)):
			I = perturbed(I, PX[j], PY[j], r)	
		# Check whether the perturbed image I is an adversarial image
		if label not in top_k_predicitons(model, [perturbed], k):
			return True
		# Update a neighborhood of pixel locations for the next round
		PX_ , PY_ = [], []
		for j in range(len(PX)):
			for k in range(-d,d+1):
				x_co = PX[j] + k
				y_co = PY[j] + k
				if x_co > 0 and x_co < I.shape[1] and y_co > 0 and y_co < Y.shape[2]:
					PX_.append(x_co)
					PY_.append(y_co)
		PX, PY = np.array(PX_), np.array(PY_)
		i += 1
	return False


if __name__ == "__main__":
	model = load_model(sys.argv[1])
