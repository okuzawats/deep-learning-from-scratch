import numpy as np
import matplotlib.pylab as plt

def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = np.sum(x * w) + b
	if tmp <= 0:
		return 0
	else:
		return 1

def NAND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5])
	b = 0.7
	tmp = np.sum(x * w) + b
	if tmp <= 0:
		return 0
	else:
		return 1

def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.2
	tmp = np.sum(x * w) + b
	if tmp <= 0:
		return 0
	else:
		return 1

def XOR(x1, x2):
	s1 = NAND(x1, x2)
	s2 = OR(x1, x2)
	y = AND(s1, s2)
	return y

def step_function(x):
	y = x > 0
	return y.astype(np.int)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c) # オーバーフロー対策
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y

if __name__ == '__main__':
	x = np.arange(-5.0, 5.0, 0.1)
	y = sigmoid(x)
	plt.plot(x, y)
	plt.ylim(-0.1, 1.1)
	plt.show()
