use vexus::NeuralNetwork;

fn normalize(x: f32, min: f32, max: f32) -> f32 {
    (x - min) / (max - min)
}

fn denormalize(x: f32, min: f32, max: f32) -> f32 {
    x * (max - min) + min
}

fn main() {
    let mut nn = if let Ok(nn) = NeuralNetwork::from_file("square_function.json") {
        nn
    } else {
        NeuralNetwork::new(vec![1, 10, 1], 0.01)
    };
    // Create a neural network with 1 input, one hidden layer of 10 neurons, and 1 output

    // Generate training data: f(x) = x^2 for x in [-1, 1]
    let training_data: Vec<(Vec<f32>, Vec<f32>)> = (-100..=100)
        .map(|i| {
            let x = i as f32 / 100.0;
            let y = x * x;
            (vec![normalize(x, -1.0, 1.0)], vec![normalize(y, 0.0, 1.0)])
        })
        .collect();

    // Train the network
    println!("Training...");
    for epoch in 0..100000 {
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
    let test_points = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    for x in test_points {
        nn.forward(vec![normalize(x, -1.0, 1.0)]);
        let predicted = denormalize(nn.get_outputs()[0], 0.0, 1.0);
        println!(
            "x = {:.3}, x^2 = {:.3}, predicted = {:.3}, error = {:.3}",
            x,
            x * x,
            predicted,
            ((x * x) - predicted).abs()
        );
    }
    nn.save_to_file("square_function.json").expect("your mom");
}
