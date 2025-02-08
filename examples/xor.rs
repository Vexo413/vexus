use vexus::NeuralNetwork;

fn main() {
    // Create a neural network with 2 inputs, one hidden layer of 4 neurons, and 1 output
    let mut nn = NeuralNetwork::new(vec![2, 4, 1], 0.1);

    // Training data for XOR
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Train the network
    for _ in 0..100000 {
        for (inputs, expected) in &training_data {
            nn.forward(inputs.clone());
            let outputs = nn.get_outputs();
            let errors = vec![expected[0] - outputs[0]];
            nn.backwards(errors);
        }
    }

    // Test the network
    for (inputs, expected) in &training_data {
        nn.forward(inputs.clone());
        let outputs = nn.get_outputs();
        println!(
            "Input: {:?}, Expected: {:?}, Got: {:.4}",
            inputs, expected[0], outputs[0]
        );
    }
}
