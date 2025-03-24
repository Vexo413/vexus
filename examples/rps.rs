use rand::Rng;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;
use vexus::NeuralNetwork;

const MODEL_FILE_1: &str = "rps_model_1.json";
const MODEL_FILE_2: &str = "rps_model_2.json";
const HISTORY_LENGTH: usize = 32; // Number of previous moves to consider
const TRAINING_ITERATIONS: usize = 10; // Training iterations per move
const AI_VS_AI_GAMES: usize = 2000; // Number of games to play in AI vs AI mode
const AI_VS_AI_DELAY_MS: u64 = 0; // Delay between moves in AI vs AI mode

#[derive(Debug, PartialEq, Clone, Copy)]
enum Move {
    Rock = 0,
    Paper = 1,
    Scissors = 2,
}

impl Move {
    fn from_str(s: &str) -> Option<Move> {
        match s.to_lowercase().as_str() {
            "r" | "rock" => Some(Move::Rock),
            "p" | "paper" => Some(Move::Paper),
            "s" | "scissors" => Some(Move::Scissors),
            _ => None,
        }
    }

    fn from_index(idx: usize) -> Move {
        match idx % 3 {
            0 => Move::Rock,
            1 => Move::Paper,
            2 => Move::Scissors,
            _ => unreachable!(),
        }
    }

    fn beats(&self, other: &Move) -> bool {
        match (self, other) {
            (Move::Rock, Move::Scissors) => true,
            (Move::Paper, Move::Rock) => true,
            (Move::Scissors, Move::Paper) => true,
            _ => false,
        }
    }

    fn random() -> Move {
        let mut rng = rand::rng();
        match rng.random_range(0..3) {
            0 => Move::Rock,
            1 => Move::Paper,
            _ => Move::Scissors,
        }
    }

    fn counter(&self) -> Move {
        match self {
            Move::Rock => Move::Paper,
            Move::Paper => Move::Scissors,
            Move::Scissors => Move::Rock,
        }
    }

    fn to_string(&self) -> &'static str {
        match self {
            Move::Rock => "Rock",
            Move::Paper => "Paper",
            Move::Scissors => "Scissors",
        }
    }

    fn to_input_vec(&self) -> Vec<f32> {
        let mut result = vec![0.0, 0.0, 0.0];
        result[*self as usize] = 1.0;
        result
    }
}

struct MovePredictor {
    nn: NeuralNetwork,
    player_history: Vec<Move>,
    initialized: bool,
    model_file: String,
}

impl MovePredictor {
    fn new(model_file: &str) -> Self {
        // Try to load an existing model or create a new one
        let nn = match NeuralNetwork::from_file(model_file) {
            Ok(model) => {
                println!("Loaded neural network from file: {}", model_file);
                model
            }
            Err(_) => {
                println!("Creating a new neural network model.");
                // Input layer: HISTORY_LENGTH * 3 (one-hot encoding of Rock, Paper, Scissors)
                // Hidden layer: 12 neurons
                // Output layer: 3 neurons (probabilities for Rock, Paper, Scissors)
                NeuralNetwork::new(vec![HISTORY_LENGTH * 3, 32, 32, 16, 3], 0.1)
            }
        };

        MovePredictor {
            nn,
            player_history: Vec::new(),
            initialized: false,
            model_file: model_file.to_string(),
        }
    }

    fn record_move(&mut self, player_move: Move) {
        self.player_history.push(player_move);

        // Keep only the most recent moves
        if self.player_history.len() > HISTORY_LENGTH * 2 {
            self.player_history =
                self.player_history[self.player_history.len() - HISTORY_LENGTH * 2..].to_vec();
        }

        // Mark as initialized once we have enough history
        if self.player_history.len() >= HISTORY_LENGTH {
            self.initialized = true;
        }
    }

    fn train(&mut self) {
        if !self.initialized || self.player_history.len() < HISTORY_LENGTH + 1 {
            return;
        }

        // Train on sequences in the history
        for i in 0..self.player_history.len() - HISTORY_LENGTH {
            let inputs = self.history_to_input(&self.player_history[i..i + HISTORY_LENGTH]);
            let target = self.player_history[i + HISTORY_LENGTH].to_input_vec();

            // Train multiple times on each sequence to reinforce learning
            for _ in 0..TRAINING_ITERATIONS {
                self.nn.forward(inputs.clone());
                let outputs = self.nn.get_outputs();

                // Calculate errors (expected - actual)
                let errors = target
                    .iter()
                    .zip(outputs.iter())
                    .map(|(t, o)| t - o)
                    .collect();

                self.nn.backwards(errors);
            }
        }
    }

    fn predict_next_move(&mut self) -> Move {
        if !self.initialized || self.player_history.len() < HISTORY_LENGTH {
            return Move::random();
        }

        // Get the last HISTORY_LENGTH moves
        let recent_history = &self.player_history[self.player_history.len() - HISTORY_LENGTH..];
        let inputs = self.history_to_input(recent_history);

        // Forward pass through the neural network
        self.nn.forward(inputs);
        let outputs = self.nn.get_outputs();

        // Find the move with highest probability
        let mut max_idx = 0;
        let mut max_val = outputs[0];

        for (i, &val) in outputs.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Return the move corresponding to the highest output
        Move::from_index(max_idx)
    }

    fn make_move(&mut self) -> Move {
        // For AI gameplay: predict the next move and choose a strategic response
        if !self.initialized || self.player_history.len() < HISTORY_LENGTH {
            // When not enough history, pick a random move
            return Move::random();
        }

        // More complex strategy than just counter-picking:
        // Occasionally be random to prevent being too predictable
        let mut rng = rand::rng();
        if rng.random_bool(0.2) {
            // 20% chance of random move
            return Move::random();
        }

        // Otherwise, predict and counter
        let predicted_next = self.predict_next_move();
        predicted_next.counter()
    }

    fn history_to_input(&self, history: &[Move]) -> Vec<f32> {
        let mut inputs = Vec::with_capacity(history.len() * 3);
        for &m in history {
            inputs.extend_from_slice(&m.to_input_vec());
        }
        inputs
    }

    fn save_model(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.nn.save_to_file(&self.model_file)?;
        Ok(())
    }
}

enum GameMode {
    PlayerVsAI,
    AIVsAI,
}

fn get_game_mode() -> GameMode {
    loop {
        println!("=== ROCK PAPER SCISSORS with AI ===");
        println!("Select game mode:");
        println!("1. Player vs AI");
        println!("2. AI vs AI");
        print!("Enter your choice (1-2): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        match input.trim() {
            "1" => return GameMode::PlayerVsAI,
            "2" => return GameMode::AIVsAI,
            _ => {
                println!("Invalid choice! Please enter 1 or 2.");
                println!();
            }
        }
    }
}

fn player_vs_ai_mode() {
    println!("\n=== PLAYER VS AI MODE ===");
    println!("Enter 'q' to quit at any time");

    let mut player_score = 0;
    let mut computer_score = 0;
    let mut predictor = MovePredictor::new(MODEL_FILE_1);

    loop {
        // Get player's move
        print!("Enter your move (r)ock, (p)aper, (s)cissors: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let input = input.trim();
        if input == "q" || input == "quit" {
            break;
        }

        let player_move = match Move::from_str(input) {
            Some(m) => m,
            None => {
                println!("Invalid move! Please enter 'r', 'p', or 's'.");
                continue;
            }
        };

        // Computer predicts and makes its move
        let predicted_move = predictor.predict_next_move();
        let computer_move = predicted_move.counter();

        // Record the player's move for future predictions
        predictor.record_move(player_move);

        // Train the neural network with the updated history
        predictor.train();

        println!("You chose: {}", player_move.to_string());
        println!("Computer chose: {}", computer_move.to_string());

        // Determine the winner
        if player_move == computer_move {
            println!("It's a tie!");
        } else if player_move.beats(&computer_move) {
            println!("You win this round!");
            player_score += 1;
        } else {
            println!("Computer wins this round!");
            computer_score += 1;
        }

        println!(
            "Score - You: {}, Computer: {}",
            player_score, computer_score
        );
        println!();

        // Save the model occasionally
        if (player_score + computer_score) % 5 == 0 {
            if let Err(e) = predictor.save_model() {
                eprintln!("Failed to save model: {}", e);
            }
        }
    }

    // Save the final model
    if let Err(e) = predictor.save_model() {
        eprintln!("Failed to save model: {}", e);
    }

    println!("\nFinal Score:");
    println!("You: {}", player_score);
    println!("Computer: {}", computer_score);

    if player_score > computer_score {
        println!("Congratulations! You won the game!");
    } else if player_score < computer_score {
        println!("Better luck next time! Computer won the game.");
    } else {
        println!("It's a tie game!");
    }
}

fn ai_vs_ai_mode() {
    println!("\n=== AI VS AI MODE ===");
    println!(
        "The AIs will play {} games against each other",
        AI_VS_AI_GAMES
    );
    println!("Press Ctrl+C to stop at any time");
    println!();

    let mut ai1 = MovePredictor::new(MODEL_FILE_1);
    let mut ai2 = MovePredictor::new(MODEL_FILE_2);

    let mut ai1_score = 0;
    let mut ai2_score = 0;
    let mut ties = 0;

    println!("Game starting...");

    // Initialize with some random moves to get history
    for _ in 0..HISTORY_LENGTH {
        let random_move = Move::random();
        ai1.record_move(random_move);
        ai2.record_move(random_move);
    }

    for game in 1..=AI_VS_AI_GAMES {
        // AI 1 makes a move
        let ai1_move = ai1.make_move();

        // AI 2 makes a move
        let ai2_move = ai2.make_move();

        // Record each other's moves
        ai1.record_move(ai2_move);
        ai2.record_move(ai1_move);

        // Train both AIs
        ai1.train();
        ai2.train();

        // Determine winner
        println!(
            "Game {}: AI1 chose {}, AI2 chose {}",
            game,
            ai1_move.to_string(),
            ai2_move.to_string()
        );

        if ai1_move == ai2_move {
            println!("Game {}: It's a tie!", game);
            ties += 1;
        } else if ai1_move.beats(&ai2_move) {
            println!("Game {}: AI1 wins!", game);
            ai1_score += 1;
        } else {
            println!("Game {}: AI2 wins!", game);
            ai2_score += 1;
        }

        println!(
            "Current score - AI1: {}, AI2: {}, Ties: {}",
            ai1_score, ai2_score, ties
        );
        println!();

        // Save models occasionally
        if game % 10 == 0 {
            if let Err(e) = ai1.save_model() {
                eprintln!("Failed to save AI1 model: {}", e);
            }
            if let Err(e) = ai2.save_model() {
                eprintln!("Failed to save AI2 model: {}", e);
            }
        }

        // Delay to make it easier to follow
        thread::sleep(Duration::from_millis(AI_VS_AI_DELAY_MS));
    }

    // Save final models
    if let Err(e) = ai1.save_model() {
        eprintln!("Failed to save AI1 model: {}", e);
    }
    if let Err(e) = ai2.save_model() {
        eprintln!("Failed to save AI2 model: {}", e);
    }

    println!("\nFinal Score after {} games:", AI_VS_AI_GAMES);
    println!("AI1: {}", ai1_score);
    println!("AI2: {}", ai2_score);
    println!("Ties: {}", ties);

    if ai1_score > ai2_score {
        println!("AI1 won the tournament!");
    } else if ai1_score < ai2_score {
        println!("AI2 won the tournament!");
    } else {
        println!("The tournament ended in a tie!");
    }
}

fn main() {
    let game_mode = get_game_mode();

    match game_mode {
        GameMode::PlayerVsAI => player_vs_ai_mode(),
        GameMode::AIVsAI => ai_vs_ai_mode(),
    }
}
