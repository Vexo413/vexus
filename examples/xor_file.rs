use std::fs;
use vexus::NeuralNetwork;

fn main() {
    let file_path = "xor_model.json";

    // Try to load the neural network from a file, or create a new one if the file does not exist
    let mut nn = if let Ok(nn) = NeuralNetwork::from_file(file_path) {
        println!("Loaded neural network from file.");
        nn
    } else {
        println!("Creating a new neural network.");
        NeuralNetwork::new(vec![2, 4, 1], 0.1)
    };

    // Training data for XOR
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Train the network
    for _ in 0..1000000 {
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

    // Save the trained neural network to a file
    if let Err(e) = nn.save_to_file(file_path) {
        eprintln!("Failed to save neural network to file: {}", e);
    } else {
        println!("Neural network saved to file.");
    }
}
