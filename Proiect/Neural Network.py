import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def relu(x):
  # ReLu function f(x) = 0 if x < 0, x otherwise
  var = x
  if x < 0:
    var = 0
  return var

def deriv_relu(x):
  fx = relu(x)
  return fx * (1 - fx)

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    self.w9 = np.random.normal()
    self.w10 = np.random.normal()
    self.w11 = np.random.normal()
    self.w12 = np.random.normal()
    self.w13 = np.random.normal()
    self.w14 = np.random.normal()


    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()  # output
    self.b4 = np.random.normal()  # h3
    self.b5 = np.random.normal()  # h4
    self.b6 = np.random.normal()  # h5

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = relu(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = relu(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    h3 = relu(self.w7 * x[0] + self.w8 * x[1] + self.b4)
    h4 = relu(self.w5 * h1 + self.w6 * h2 + self.w9 * h3 + self.b5)
    h5 = relu(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b6)
    o1 = relu(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = relu(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = relu(sum_h2)

        sum_h3 = self.w7 * x[0] + self.w8 * x[1] +self.b4
        h3 = relu(sum_h3)

        sum_h4 = self.w5 * h1 + self.w6 * h2 + self.w9 * h3 + self.b5
        h4 = relu(sum_h4)

        sum_h5 = self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b6
        h5 = relu(sum_h5)

        sum_o1 = self.w13 * h4 + self.w14 * h5 + self.b3
        o1 = relu(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w13 = h4 * deriv_relu(sum_o1)
        d_ypred_d_w14 = h5 * deriv_relu(sum_o1)
        d_ypred_d_b3 = deriv_relu(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_relu(sum_h4) * self.w10 * deriv_relu(sum_h5)
        d_ypred_d_h2 = self.w6 * deriv_relu(sum_h4) * self.w11 * deriv_relu(sum_h5)
        d_ypred_d_h3 = self.w9 * deriv_relu(sum_h4) * self.w12 * deriv_relu(sum_h5)

        d_ypred_d_h4 = self.w13 * deriv_relu(sum_o1)
        d_ypred_d_h5 = self.w14 * deriv_relu(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_relu(sum_h1)
        d_h1_d_w2 = x[1] * deriv_relu(sum_h1)
        d_h1_d_b1 = deriv_relu(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_relu(sum_h2)
        d_h2_d_w4 = x[1] * deriv_relu(sum_h2)
        d_h2_d_b2 = deriv_relu(sum_h2)

        # Neuron h3
        d_h3_d_w7 = x[0] * deriv_relu(sum_h3)
        d_h3_d_w8 = x[1] * deriv_relu(sum_h3)
        d_h3_d_b4 = deriv_relu(sum_h3)

        # Neuron h4
        d_h4_d_w5 = deriv_relu(sum_h1) * deriv_relu(sum_h4)
        d_h4_d_w6 = deriv_relu(sum_h2) * deriv_relu(sum_h4)
        d_h4_d_w9 = deriv_relu(sum_h3) * deriv_relu(sum_h4)
        d_h4_d_b5 = deriv_relu(sum_h4)

        # Neuron h5
        d_h5_d_w10 = deriv_relu(sum_h1) * deriv_relu(sum_h5)
        d_h5_d_w11 = deriv_relu(sum_h2) * deriv_relu(sum_h5)
        d_h5_d_w12 = deriv_relu(sum_h3) * deriv_relu(sum_h5)
        d_h5_d_b6 = deriv_relu(sum_h5)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron h3
        self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w7
        self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w8
        self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b4

        # Neuron h4
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w6
        self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w9
        self.b5 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_b5

        # Neuron h5
        self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w10
        self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w11
        self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w12
        self.b5 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_b6


        # Neuron o1
        self.w13 -= learn_rate * d_L_d_ypred * d_ypred_d_w13
        self.w14 -= learn_rate * d_L_d_ypred * d_ypred_d_w14
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
# data = np.array([
#  [-2, -1],  # Alice
#  [25, 6],   # Bob
#  [17, 4],   # Charlie
#  [-15, -6], # Diana
#])


woman_height = np.round(np.random.normal(1, 10, 2), 0)
woman_weight = np.round(np.random.normal(1, 10, 2), 0)
man_height = np.round(np.random.normal(5, 15, 2), 0)
man_weight = np.round(np.random.normal(5, 15, 2), 0)
np_woman = np.column_stack((woman_height, woman_weight))
np_man = np.column_stack((man_height, man_weight))

data = np.array([np_woman[0], np_man[0], np_man[1], np_woman[1]])

print(data)
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M