# Vexus

A neural network builder and trainer to make your own AI models.

## Features

- Feed-forward neural network
- Backpropagation learning
- Configurable layer sizes
- Sigmoid activation function

## Installation

Add this to your `Cargo.toml`:

```ssh
cargo add vexus
```

## Quick Start

### XOR Predictor

```rust
use neural_network::NeuralNetwork;

fn main() {
    // Create a neural network with:
    // - 2 input neurons
    // - 4 hidden neurons
    // - 1 output neuron
    let mut nn = NeuralNetwork::new(vec![2, 4, 1], 0.1);

    // Training data for XOR function
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Train the network
    for _ in 0..10000 {
        for (inputs, expected) in &training_data {
            nn.forward(inputs.clone());
            let outputs = nn.get_outputs();
            let errors = vec![expected[0] - outputs[0]];
            nn.backwards(errors);
        }
    }

    // Test the network
    nn.forward(vec![1.0, 0.0]);
    println!("1 XOR 0 = {:.2}", nn.get_outputs()[0]); // Should be close to 1.0
}
```

### Run Examples

```ssh
cargo run --example xor

cargo run --example sine_waves

```

## Todo

- [ ] Add mutation
- [ ] Implement different activation functions
  - [ ] ReLU
  - [ ] Tanh
  - [x] Sigmoid
- [ ] Add save/load functionality
- [ ] Add documentation
- [ ] Add more examples


## License

MIT
