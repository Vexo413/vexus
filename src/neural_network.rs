use rand::distr::{Distribution, Uniform};
use rand::rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    weights: Vec<Vec<Vec<f32>>>,  // Layer -> Input -> Output
    biases: Vec<Vec<f32>>,        // Layer -> Output
    layer_inputs: Vec<Vec<f32>>,  // Layer -> Input
    layer_outputs: Vec<Vec<f32>>, // Layer -> Output
    layer_errors: Vec<Vec<f32>>,  // Layer -> Output
    layer_sizes: Vec<usize>,      // Number of neurons in each layer
    learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f32) -> Self {
        let mut rng = rng();
        let between = Uniform::try_from(-1.0..1.0).unwrap();
        let num_layers = layer_sizes.len() - 1; // -1 because we need pairs of layers

        let mut weights = Vec::with_capacity(num_layers);
        let mut biases = Vec::with_capacity(num_layers);
        let mut layer_inputs = Vec::with_capacity(num_layers);
        let mut layer_outputs = Vec::with_capacity(num_layers);
        let mut layer_errors = Vec::with_capacity(num_layers);

        // Initialize layers
        for i in 0..num_layers {
            let num_inputs = layer_sizes[i];
            let num_outputs = layer_sizes[i + 1];

            // Initialize weights
            let layer_weights: Vec<Vec<f32>> = (0..num_inputs)
                .map(|_| (0..num_outputs).map(|_| between.sample(&mut rng)).collect())
                .collect();
            weights.push(layer_weights);

            // Initialize biases
            let layer_biases: Vec<f32> =
                (0..num_outputs).map(|_| between.sample(&mut rng)).collect();
            biases.push(layer_biases);

            // Initialize storage vectors
            layer_inputs.push(vec![0.0; num_inputs]);
            layer_outputs.push(vec![0.0; num_outputs]);
            layer_errors.push(vec![0.0; num_outputs]);
        }

        NeuralNetwork {
            weights,
            biases,
            layer_inputs,
            layer_outputs,
            layer_errors,
            layer_sizes,
            learning_rate,
        }
    }
    pub fn from_file(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let nn = serde_json::from_reader(reader)?;
        Ok(nn)
    }

    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    pub fn forward(&mut self, inputs: Vec<f32>) {
        // First layer
        self.layer_inputs[0] = inputs.clone();
        for i in 0..self.layer_sizes[1] {
            self.layer_outputs[0][i] = 0.0;
            for j in 0..self.layer_sizes[0] {
                self.layer_outputs[0][i] += inputs[j] * self.weights[0][j][i];
            }
            self.layer_outputs[0][i] += self.biases[0][i];
            self.layer_outputs[0][i] = self.sigmoid(self.layer_outputs[0][i]);
        }

        // Hidden layers
        for layer in 1..self.weights.len() {
            self.layer_inputs[layer] = self.layer_outputs[layer - 1].clone();
            for i in 0..self.layer_sizes[layer + 1] {
                self.layer_outputs[layer][i] = 0.0;
                for j in 0..self.layer_sizes[layer] {
                    self.layer_outputs[layer][i] +=
                        self.layer_outputs[layer - 1][j] * self.weights[layer][j][i];
                }
                self.layer_outputs[layer][i] += self.biases[layer][i];
                self.layer_outputs[layer][i] = self.sigmoid(self.layer_outputs[layer][i]);
            }
        }
    }

    pub fn backwards(&mut self, errors: Vec<f32>) {
        let last_layer = self.weights.len() - 1;

        // Output layer errors
        for i in 0..self.layer_sizes[last_layer + 1] {
            self.layer_errors[last_layer][i] =
                errors[i] * self.sigmoid_derivative(self.layer_outputs[last_layer][i]);
        }

        // Hidden layer errors
        for layer in (0..last_layer).rev() {
            for j in 0..self.layer_sizes[layer + 1] {
                self.layer_errors[layer][j] = 0.0;
                for k in 0..self.layer_sizes[layer + 2] {
                    self.layer_errors[layer][j] +=
                        self.layer_errors[layer + 1][k] * self.weights[layer + 1][j][k];
                }
                self.layer_errors[layer][j] *=
                    self.sigmoid_derivative(self.layer_outputs[layer][j]);
            }
        }

        // Update weights and biases
        for layer in 0..self.weights.len() {
            for j in 0..self.layer_sizes[layer + 1] {
                for i in 0..self.layer_sizes[layer] {
                    self.weights[layer][i][j] += self.learning_rate
                        * self.layer_errors[layer][j]
                        * self.layer_inputs[layer][i];
                }
                self.biases[layer][j] += self.learning_rate * self.layer_errors[layer][j];
            }
        }
    }
    pub fn get_outputs(&self) -> Vec<f32> {
        // New public method
        self.layer_outputs.last().unwrap().clone()
    }

    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(&self, x: f32) -> f32 {
        x * (1.0 - x)
    }
}
