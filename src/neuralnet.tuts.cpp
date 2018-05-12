// neuralnet.tuts.cpp
// https://www.youtube.com/watch?v=KkwX7FkLfug
/*
* A two-dimensional vector of vector of Nuerons
* could be declared:
* vector< vector<Neuron> > m_layersOfNeurons;
*/

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct Connection {
  double weight;
  double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ******************* class Neuron ******************

class Neuron {
  public:
     // inititiate the constructor Neuron(...)
    Neuron(unsigned numOutputs, unsigned myIndex);
    double setOutputVal(double val) const { return m_outputVal; }
    double getOutputVal (void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

  private:
    static double eta;  // [0.0 .. 1.0] overall net training rate
    static double alpha; // [0.0 .. n] multiplier of last weight change (mommentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight (void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer);
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
  for (unsigned c = 0; c < numOutputs; ++c) {
    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight = randomWeight();
  }

  m_myIndex = myIndex;
}

double Neuron::eta = 0.15; // overall net learing rate
double Neuron::alpha = 0.5; // momentum multiplier of last deltaWeight, [0.0 .. n]


void Neuron::updateInputWeights(Layer &prevLayer) {
  // the wieghts to be updated are in the connection container
  // in the neurons in the preceding layer

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

    double newDeltaWeight =
      // individual input, magnified by the gradient and train rate:
      eta
      * neuron_getOutputVal()
      *m_gradient
      // also add momment  = a fraction of the previous delta weight
      + alpha
      * oldDeltaWeight;

    neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}

double Neuron::sumDOW(const Layer &nextLayer) {
  double sum = 0.0;
  // sum our contributions of the errors at the nodes we feed
  for (unsigned n = 0; n < nextLayer.size() -1; ++n) {
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }
  return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
  double dow = sumDOW(nextLayer);
  m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);

}

void Neuron::calcOutputGradients(double targetVal) {
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) {
  // tangent fucntion = output rang [-1.0 .. 1.0]
  return tanh(x);
};

double Neuron::transferFunctionDerivative(double x) {
  // tanh derivative
  return 1.0 - x * x;
};

void Neuron::feedForward(const Layer &prevLayer) {
  double sum = 0.0;
  // sum the previous layers output (which are our inputs)
  // include the bias node from the previous layer
  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
  }
  m_outputVal = Neuron::transferFunction(sum);
};

// ******************* class Net ******************
class Net {
  public:
    // initiate the constructor Net(...)
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(const vector<double> &resultVals) const;

  private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

Net::Net(const vector<unsigned> &topology) {
  unsigned numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
    m_layers.push_back(Layer());
    unsigned numOutputs = layerNum == topology.size() - 1? 0 : topology[layerNum + 1];
    // we have made a new Layer, now fill it with neurons. and
    // add a bias neuron to the layer:
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));
      cout << "Made a Neuron!!" << endl;
    }
    // Force the bias node's out value to 1.0
    // It's the last neuron created above
    m_layers.back().back().setOutputVal(1.0);
  }
};

void Net::getResults(const vector<double> &resultVals) const {
  // clears out the container
  resultVals.clear();
  for (unsigned n = 0; n < m_layers.back().size() -1; ++n) {
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }

}
// define member function
void Net::feedForward(const vector<double> &inputVals) {
  assert(inputVals.size() == m_layers[0].size() - 1);
  // Assigne (latch) the input values into the input neurons
  for (unsigned i = 0; i < inputVals.size(); ++i) {
    m_layers[0][i].setOutputVal(inputVals[i]);
  }
  // forward propagate
  for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
    Layer &prevLayer = m_layers[layerNum - 1];
    for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
      m_layers[layerNum][n].feedForward(prevLayer);
    }
  }
};

void Net::backProp(const vector<double> &targetVals) {
  // calculate overall net error (RMS of output neuron errors)
  // for all layers from outputs to first hidden layer.
  // update connection weights
  Layer &outputLayer = m_layers.back();
  m_error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1; // get average error squared
  m_error = sqrt(m_error);

  // calculate output layer gradients
  m_recentAverageError =
    (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
    (m_recentAverageSmoothingFactor + 1.0);

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  // calculate gradients on hidden layers
  for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
    Layer &hiddenLayer = m_layers[layerNum];
    Layer &nextLayer = m_layers[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size() - 1; ++n) {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }

  // update the weights
  for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
    Layer &layer = m_layers[layerNum];
    Layer &prevLayer = m_layers[layerNum - 1];

    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }

};

int main () {
  // give some structure
  // { 3, 2, 1 }        ø
  //@ 3 input layers    ø ø ø
  //@ 2 hidden layers   ø ø
  //@ 1 output layer
  vector<unsigned> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(1);

  Net myNet(topology);
  // use vector like a variable length array
  // in-order to train it we need to be able to call some member functions
  std::vector<double> inputVals;
  myNet.feedForward(inputVals);
  // back propegation
  std::vector<double> targetVals;
  myNet.backProp(targetVals);

  std::vector<double> resultVals;
  myNet.getResults(resultVals);
}
