use std::f32::consts::PI;
use vexus::NeuralNetwork;

fn normalize(x: f32, min: f32, max: f32) -> f32 {
    (x - min) / (max - min)
}

fn denormalize(x: f32, min: f32, max: f32) -> f32 {
    x * (max - min) + min
}

fn main() {
    // Create a network with 1 input, two hidden layers, and 1 output
    // Larger architecture to handle the complexity of sine function
    let mut nn = NeuralNetwork::new(vec![1, 32, 32, 1], 0.005);

    // Generate training data: sin(x) for x in [0, 2π]
    let training_data: Vec<(Vec<f32>, Vec<f32>)> = (0..200)
        .map(|i| {
            let x = (i as f32) * 2.0 * PI / 200.0;
            let normalized_x = normalize(x, 0.0, 2.0 * PI);
            let normalized_sin = normalize(x.sin(), -1.0, 1.0);
            (vec![normalized_x], vec![normalized_sin])
        })
        .collect();

    // Train the network
    println!("Training...");
    for epoch in 0..20000 {
        let mut total_error = 0.0;
        for (input, expected) in &training_data {
            nn.forward(input.clone());
            let output = nn.get_outputs();
            let error = expected[0] - output[0];
            total_error += error * error;
            nn.backwards(vec![error]);
        }

        if epoch % 1000 == 0 {
            println!(
                "Epoch {}: MSE = {:.6}",
                epoch,
                total_error / training_data.len() as f32
            );
        }
    }

    // Test the network
    println!("\nTesting...");
    let test_points = vec![0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI];
    for x in test_points {
        let normalized_x = normalize(x, 0.0, 2.0 * PI);
        nn.forward(vec![normalized_x]);
        let predicted = denormalize(nn.get_outputs()[0], -1.0, 1.0);
        println!(
            "x = {:.3}, sin(x) = {:.3}, predicted = {:.3}, error = {:.3}",
            x,
            x.sin(),
            predicted,
            (x.sin() - predicted).abs()
        );
    }
}
