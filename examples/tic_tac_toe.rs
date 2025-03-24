use rand::prelude::*;
use std::fs;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;
use vexus::NeuralNetwork;

const MODEL_FILE: &str = "tictactoe_nn.json";
const CLEAR_SCREEN: &str = "\x1B[2J\x1B[1;1H";
const RED: &str = "\x1B[31m";
const GREEN: &str = "\x1B[32m";
const YELLOW: &str = "\x1B[33m";
const BLUE: &str = "\x1B[34m";
const PURPLE: &str = "\x1B[35m";
const CYAN: &str = "\x1B[36m";
const BOLD: &str = "\x1B[1m";
const RESET: &str = "\x1B[0m";
const MODEL_X_FILE: &str = "tictactoe_x_model.json";
const MODEL_O_FILE: &str = "tictactoe_o_model.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Empty,
    X,
    O,
}

impl Cell {
    fn to_colored_string(&self) -> String {
        match self {
            Cell::Empty => " ".to_string(),
            Cell::X => format!("{}{}{}", GREEN, "X", RESET),
            Cell::O => format!("{}{}{}", RED, "O", RESET),
        }
    }

    fn opponent(&self) -> Self {
        match self {
            Cell::X => Cell::O,
            Cell::O => Cell::X,
            Cell::Empty => Cell::Empty,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameState {
    InProgress,
    XWins,
    OWins,
    Draw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Player {
    Human,
    AI,
}

#[derive(Clone)]
struct Board {
    cells: [Cell; 9],
}

impl Board {
    fn new() -> Self {
        Self {
            cells: [Cell::Empty; 9],
        }
    }

    fn make_move(&mut self, index: usize, cell: Cell) -> bool {
        if index < 9 && self.cells[index] == Cell::Empty {
            self.cells[index] = cell;
            true
        } else {
            false
        }
    }

    fn is_full(&self) -> bool {
        self.cells.iter().all(|&cell| cell != Cell::Empty)
    }

    fn get_empty_cells(&self) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(i, &cell)| if cell == Cell::Empty { Some(i) } else { None })
            .collect()
    }

    fn check_winner(&self) -> GameState {
        for i in 0..3 {
            if self.cells[i * 3] != Cell::Empty
                && self.cells[i * 3] == self.cells[i * 3 + 1]
                && self.cells[i * 3] == self.cells[i * 3 + 2]
            {
                return match self.cells[i * 3] {
                    Cell::X => GameState::XWins,
                    Cell::O => GameState::OWins,
                    _ => unreachable!(),
                };
            }
        }

        for i in 0..3 {
            if self.cells[i] != Cell::Empty
                && self.cells[i] == self.cells[i + 3]
                && self.cells[i] == self.cells[i + 6]
            {
                return match self.cells[i] {
                    Cell::X => GameState::XWins,
                    Cell::O => GameState::OWins,
                    _ => unreachable!(),
                };
            }
        }

        if self.cells[0] != Cell::Empty
            && self.cells[0] == self.cells[4]
            && self.cells[0] == self.cells[8]
        {
            return match self.cells[0] {
                Cell::X => GameState::XWins,
                Cell::O => GameState::OWins,
                _ => unreachable!(),
            };
        }

        if self.cells[2] != Cell::Empty
            && self.cells[2] == self.cells[4]
            && self.cells[2] == self.cells[6]
        {
            return match self.cells[2] {
                Cell::X => GameState::XWins,
                Cell::O => GameState::OWins,
                _ => unreachable!(),
            };
        }

        if self.is_full() {
            return GameState::Draw;
        }

        GameState::InProgress
    }

    fn to_nn_input(&self, player_symbol: Cell) -> Vec<f32> {
        let mut input = Vec::with_capacity(18);

        for &cell in &self.cells {
            input.push(if cell == player_symbol { 1.0 } else { 0.0 });
        }

        let opponent_symbol = player_symbol.opponent();
        for &cell in &self.cells {
            input.push(if cell == opponent_symbol { 1.0 } else { 0.0 });
        }

        input
    }
}

struct NeuralNetworkAI {
    nn: NeuralNetwork,
    training_data: Vec<(Vec<f32>, Vec<f32>)>,
    symbol: Cell,
}

impl NeuralNetworkAI {
    fn new(symbol: Cell) -> Self {
        let nn = match NeuralNetwork::from_file(MODEL_FILE) {
            Ok(model) => {
                println!("Loaded neural network from file: {}", MODEL_FILE);
                model
            }
            Err(_) => {
                println!("Creating a new neural network model.");
                NeuralNetwork::new(vec![18, 36, 36, 36, 9], 0.01)
            }
        };

        Self {
            nn,
            training_data: Vec::new(),
            symbol,
        }
    }

    fn get_move(&mut self, board: &Board) -> usize {
        let empty_cells = board.get_empty_cells();

        if empty_cells.len() == 1 {
            return empty_cells[0];
        }

        let input = board.to_nn_input(self.symbol);

        self.nn.forward(input);
        let output = self.nn.get_outputs();

        let mut valid_moves: Vec<(usize, f32)> =
            empty_cells.iter().map(|&idx| (idx, output[idx])).collect();

        valid_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut rng = rand::rng();
        if rng.random::<f32>() < 0.1 {
            *empty_cells.choose(&mut rng).unwrap()
        } else {
            valid_moves[0].0
        }
    }

    fn record_game(&mut self, board_states: &Vec<(Board, usize)>, winner: GameState) {
        for (board, move_idx) in board_states {
            let current_player =
                if board.cells.iter().filter(|&&c| c != Cell::Empty).count() % 2 == 0 {
                    Cell::X
                } else {
                    Cell::O
                };

            if current_player != self.symbol {
                continue;
            }

            let input = board.to_nn_input(self.symbol);

            let mut target = vec![0.0; 9];

            let move_value = match winner {
                GameState::XWins => {
                    if self.symbol == Cell::X {
                        1.0
                    } else {
                        0.0
                    }
                }
                GameState::OWins => {
                    if self.symbol == Cell::O {
                        1.0
                    } else {
                        0.0
                    }
                }
                GameState::Draw => 0.5,
                _ => 0.1,
            };

            target[*move_idx] = move_value;

            self.training_data.push((input, target));
        }
    }

    fn train_incremental(&mut self, epochs: usize) {
        if self.training_data.len() < 3 {
            return;
        }

        let mut rng = rand::rng();
        let data_size = self.training_data.len();
        let mut indices: Vec<usize> = (0..data_size).collect();
        indices.shuffle(&mut rng);

        let training_size = data_size.min(50);
        let indices = indices[0..training_size].to_vec();

        for _ in 0..epochs {
            for &idx in &indices {
                let (input, target) = &self.training_data[idx];

                self.nn.forward(input.clone());
                let outputs = self.nn.get_outputs();

                let errors: Vec<f32> = target
                    .iter()
                    .zip(outputs.iter())
                    .map(|(t, o)| t - o)
                    .collect();

                self.nn.backwards(errors);
            }
        }

        if self.training_data.len() > 1000 {
            self.training_data = self.training_data[self.training_data.len() - 1000..].to_vec();
        }
    }
}

struct TicTacToe {
    board: Board,
    player_symbol: Cell,
    ai_symbol: Cell,
    current_turn: Player,
    ai: NeuralNetworkAI,
    games_played: usize,
    player_wins: usize,
    ai_wins: usize,
    draws: usize,
}

impl TicTacToe {
    fn new() -> Self {
        Self {
            board: Board::new(),
            player_symbol: Cell::X,
            ai_symbol: Cell::O,
            current_turn: Player::Human,
            ai: NeuralNetworkAI::new(Cell::O),
            games_played: 0,
            player_wins: 0,
            ai_wins: 0,
            draws: 0,
        }
    }

    fn reset_game(&mut self) {
        self.board = Board::new();

        if self.player_symbol == Cell::X {
            self.current_turn = Player::Human;
        } else {
            self.current_turn = Player::AI;
        }

        self.ai.symbol = self.ai_symbol;
    }

    fn display_board(&self) {
        clear_screen();
        println!(
            "{}{}TIC TAC TOE WITH NEURAL NETWORK{}{}",
            BOLD, BLUE, RESET, YELLOW
        );
        println!("----------------------------------");

        if self.player_symbol == Cell::X {
            println!("You: {}X{}  |  AI: {}O{}", GREEN, RESET, RED, RESET);
        } else {
            println!("You: {}O{}  |  AI: {}X{}", RED, RESET, GREEN, RESET);
        }

        println!(
            "\nCurrent turn: {}",
            if self.current_turn == Player::Human {
                format!("{}YOUR TURN{}", GREEN, RESET)
            } else {
                format!("{}AI'S TURN{}", RED, RESET)
            }
        );

        println!(
            "\nStats: You: {} | AI: {} | Draws: {}",
            self.player_wins, self.ai_wins, self.draws
        );

        println!("\n");
        println!(
            " {} | {} | {} ",
            self.board.cells[0].to_colored_string(),
            self.board.cells[1].to_colored_string(),
            self.board.cells[2].to_colored_string()
        );
        println!("-----------");
        println!(
            " {} | {} | {} ",
            self.board.cells[3].to_colored_string(),
            self.board.cells[4].to_colored_string(),
            self.board.cells[5].to_colored_string()
        );
        println!("-----------");
        println!(
            " {} | {} | {} ",
            self.board.cells[6].to_colored_string(),
            self.board.cells[7].to_colored_string(),
            self.board.cells[8].to_colored_string()
        );
        println!();

        println!("{}Board positions:{}", CYAN, RESET);
        println!(" 1 | 2 | 3 ");
        println!("-----------");
        println!(" 4 | 5 | 6 ");
        println!("-----------");
        println!(" 7 | 8 | 9 ");
        println!();
    }

    fn make_player_move(&mut self, position: usize) -> bool {
        if position < 1 || position > 9 {
            return false;
        }

        let index = position - 1;
        self.board.make_move(index, self.player_symbol)
    }
    fn train_specialized_ai(&mut self) {
        clear_screen();
        println!(
            "{}{}SPECIALIZED AI TRAINING{}{}",
            BOLD, PURPLE, RESET, YELLOW
        );
        println!("------------------------");
        println!("This mode creates two separate specialized neural networks:");
        println!("1. X-AI: Specialized in playing as the first player (X)");
        println!("2. O-AI: Specialized in playing as the second player (O)");
        println!();
        println!("Each AI will be trained only for its specific role,");
        println!("allowing them to develop more focused strategies.");
        println!();
        println!("How many games should be played?");
        println!("(Recommended: 1000-5000 games)");
        println!();
        print!("Enter number of games (or 0 to cancel): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if let Ok(num_games) = input.parse::<usize>() {
            if num_games > 0 {
                clear_screen();
                println!(
                    "{}{}SPECIALIZED TRAINING IN PROGRESS{}\n",
                    BOLD, PURPLE, RESET
                );

                let mut ai_x = NeuralNetworkAI::new(Cell::X);
                let mut ai_o = NeuralNetworkAI::new(Cell::O);

                let mut x_wins = 0;
                let mut o_wins = 0;
                let mut draws = 0;

                println!("\nTraining for {} games...", num_games);

                for game in 1..=num_games {
                    if game % 100 == 0 {
                        println!("Completed {} games...", game);
                    } else if game % 10 == 0 {
                        print!(".");
                        io::stdout().flush().unwrap();
                    }

                    let mut board = Board::new();
                    let mut game_history: Vec<(Board, usize)> = Vec::new();
                    let mut current_player = Cell::X;

                    while board.check_winner() == GameState::InProgress {
                        let board_before = board.clone();

                        let ai_move = if current_player == Cell::X {
                            ai_x.symbol = Cell::X;
                            ai_x.get_move(&board)
                        } else {
                            ai_o.symbol = Cell::O;
                            ai_o.get_move(&board)
                        };

                        board.make_move(ai_move, current_player);

                        game_history.push((board_before, ai_move));

                        current_player = current_player.opponent();
                    }

                    let result = board.check_winner();

                    match result {
                        GameState::XWins => x_wins += 1,
                        GameState::OWins => o_wins += 1,
                        GameState::Draw => draws += 1,
                        _ => {}
                    }

                    ai_x.symbol = Cell::X;
                    ai_x.record_game(&game_history, result);

                    ai_o.symbol = Cell::O;
                    ai_o.record_game(&game_history, result);

                    if game % 100 == 0 || game == num_games {
                        println!("\nTraining X-AI...");
                        ai_x.train_incremental(5);

                        println!("Training O-AI...");
                        ai_o.train_incremental(5);
                    }

                    if game % 1000 == 0 || game == num_games {
                        println!("Saving specialized models...");

                        if let Err(e) = ai_x.nn.save_to_file(MODEL_X_FILE) {
                            eprintln!("Failed to save X model: {}", e);
                        }

                        if let Err(e) = ai_o.nn.save_to_file(MODEL_O_FILE) {
                            eprintln!("Failed to save O model: {}", e);
                        }
                    }
                }

                println!("\n{}Training complete!{}", GREEN, RESET);
                println!("Games played: {}", num_games);
                println!(
                    "X wins: {} ({:.1}%)",
                    x_wins,
                    (x_wins as f32 / num_games as f32) * 100.0
                );
                println!(
                    "O wins: {} ({:.1}%)",
                    o_wins,
                    (o_wins as f32 / num_games as f32) * 100.0
                );
                println!(
                    "Draws: {} ({:.1}%)",
                    draws,
                    (draws as f32 / num_games as f32) * 100.0
                );

                println!("\nSaving final specialized models...");
                if let Err(e) = ai_x.nn.save_to_file(MODEL_X_FILE) {
                    eprintln!("Failed to save X model: {}", e);
                }
                if let Err(e) = ai_o.nn.save_to_file(MODEL_O_FILE) {
                    eprintln!("Failed to save O model: {}", e);
                }

                println!("\nWhich specialized model would you like to use as default?");
                println!("{}[1]{} X Model (First Player)", GREEN, RESET);
                println!("{}[2]{} O Model (Second Player)", RED, RESET);
                println!("{}[3]{} Keep Current Model", YELLOW, RESET);
                print!("Enter your choice: ");
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();

                match input.trim() {
                    "1" => {
                        self.ai.nn = ai_x.nn.clone();
                        if let Err(e) = self.ai.nn.save_to_file(MODEL_FILE) {
                            eprintln!("Failed to save default model: {}", e);
                        } else {
                            println!("X model saved as default.");
                        }
                    }
                    "2" => {
                        self.ai.nn = ai_o.nn.clone();
                        if let Err(e) = self.ai.nn.save_to_file(MODEL_FILE) {
                            eprintln!("Failed to save default model: {}", e);
                        } else {
                            println!("O model saved as default.");
                        }
                    }
                    _ => {
                        println!("Keeping current model as default.");
                    }
                }

                println!("\nPress Enter to continue...");
                io::stdout().flush().unwrap();
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
            }
        } else {
            println!("{}Invalid input. Training cancelled.{}", RED, RESET);
            thread::sleep(Duration::from_secs(1));
        }
    }

    fn make_ai_move(&mut self) {
        thread::sleep(Duration::from_millis(500));

        let move_index = self.ai.get_move(&self.board);

        self.board.make_move(move_index, self.ai_symbol);
    }

    fn display_game_over(&self, state: GameState) {
        self.display_board();

        println!("{}GAME OVER{}", BOLD, RESET);

        match state {
            GameState::XWins => {
                if self.player_symbol == Cell::X {
                    println!("{}You win! Congratulations!{}", GREEN, RESET);
                } else {
                    println!("{}AI wins!{}", RED, RESET);
                }
            }
            GameState::OWins => {
                if self.player_symbol == Cell::O {
                    println!("{}You win! Congratulations!{}", GREEN, RESET);
                } else {
                    println!("{}AI wins!{}", RED, RESET);
                }
            }
            GameState::Draw => {
                println!("{}It's a draw!{}", YELLOW, RESET);
            }
            _ => {}
        }

        println!("\nPress Enter to continue...");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
    }

    fn play_game(&mut self) {
        self.reset_game();

        let mut game_history: Vec<(Board, usize)> = Vec::new();

        loop {
            self.display_board();

            let state = self.board.check_winner();
            if state != GameState::InProgress {
                match state {
                    GameState::XWins => {
                        if self.player_symbol == Cell::X {
                            self.player_wins += 1;
                        } else {
                            self.ai_wins += 1;
                        }
                    }
                    GameState::OWins => {
                        if self.player_symbol == Cell::O {
                            self.player_wins += 1;
                        } else {
                            self.ai_wins += 1;
                        }
                    }
                    GameState::Draw => {
                        self.draws += 1;
                    }
                    _ => {}
                }

                let original_symbol = self.ai.symbol;
                self.ai.symbol = self.ai_symbol;
                self.ai.record_game(&game_history, state);

                println!("\n{}AI is learning from this game...{}", PURPLE, RESET);
                io::stdout().flush().unwrap();

                self.ai.train_incremental(5);

                self.ai.symbol = original_symbol;

                self.games_played += 1;
                self.display_game_over(state);

                if let Err(e) = self.ai.nn.save_to_file(MODEL_FILE) {
                    eprintln!("Failed to save model: {}", e);
                }

                break;
            }

            if self.current_turn == Player::Human {
                println!(
                    "{}Your turn! Enter position (1-9) or 'q' to quit: {}",
                    GREEN, RESET
                );
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input.eq_ignore_ascii_case("q") {
                    break;
                }

                if let Ok(position) = input.parse::<usize>() {
                    let board_before = self.board.clone();

                    if self.make_player_move(position) {
                        game_history.push((board_before, position - 1));

                        self.current_turn = Player::AI;
                    } else {
                        println!("{}Invalid move! Try again.{}", RED, RESET);
                        thread::sleep(Duration::from_secs(1));
                    }
                } else {
                    println!("{}Invalid input! Try again.{}", RED, RESET);
                    thread::sleep(Duration::from_secs(1));
                }
            } else {
                let board_before = self.board.clone();
                self.make_ai_move();

                let ai_move_idx = self
                    .board
                    .cells
                    .iter()
                    .enumerate()
                    .find(|(i, cell)| {
                        **cell == self.ai_symbol && board_before.cells[*i] == Cell::Empty
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                game_history.push((board_before, ai_move_idx));

                self.current_turn = Player::Human;
            }
        }
    }

    fn display_main_menu(&mut self) {
        clear_screen();
        println!(
            "{}{}TIC TAC TOE WITH NEURAL NETWORK{}{}",
            BOLD, BLUE, RESET, YELLOW
        );
        println!("----------------------------------");
        println!(
            "{}Welcome to Tic Tac Toe with a Neural Network AI!{}",
            YELLOW, RESET
        );
        println!("The AI learns from playing against itself and you!");
        println!();
        println!("{}[1]{} Play as X (First Move)", GREEN, RESET);
        println!("{}[2]{} Play as O (Second Move)", RED, RESET);
        println!("{}[3]{} Train AI (Competitive)", YELLOW, RESET);
        println!("{}[4]{} Train Specialized AIs (X & O)", PURPLE, RESET);
        println!("{}[5]{} Watch AI vs AI Game", CYAN, RESET);
        println!("{}[6]{} View Stats", BLUE, RESET);
        println!("{}[7]{} Exit", RED, RESET);
        println!();
        print!("Enter your choice: ");
        io::stdout().flush().unwrap();
    }

    fn train_ai(&mut self) {
        clear_screen();
        println!("{}{}AI TRAINING{}{}", BOLD, PURPLE, RESET, YELLOW);
        println!("------------------------");
        println!("This method creates two separate AIs that compete against each other:");
        println!("1. Both AIs start with the current neural network");
        println!("2. They play a batch of games against each other");
        println!("3. The winner becomes the new saved neural network");
        println!();
        println!("How many games should be played in each batch?");
        println!("(Recommended: 100-1000 games per batch)");
        println!();
        print!("Enter number of games per batch (or 0 to cancel): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if let Ok(games_per_batch) = input.parse::<usize>() {
            if games_per_batch > 0 {
                println!("How many batches do you want to run?");
                print!("Enter number of batches: ");
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if let Ok(num_batches) = input.parse::<usize>() {
                    if num_batches > 0 {
                        clear_screen();
                        println!(
                            "{}{}COMPETITIVE TRAINING IN PROGRESS{}\n",
                            BOLD, PURPLE, RESET
                        );

                        let mut ai_1 = NeuralNetworkAI::new(Cell::X);
                        let mut ai_2 = NeuralNetworkAI::new(Cell::O);

                        ai_1.nn = self.ai.nn.clone();
                        ai_2.nn = self.ai.nn.clone();

                        for batch in 1..=num_batches {
                            println!("Running batch {}/{}...", batch, num_batches);

                            let mut ai_1_wins = 0;
                            let mut ai_2_wins = 0;
                            let mut draws = 0;

                            for game in 1..=games_per_batch {
                                if game % 20 == 0 {
                                    print!(".");
                                    io::stdout().flush().unwrap();
                                }

                                let (a1_symbol, _) = if game % 2 == 0 {
                                    (Cell::X, Cell::O)
                                } else {
                                    (Cell::O, Cell::X)
                                };

                                let mut board = Board::new();
                                let mut current_player = Cell::X;

                                while board.check_winner() == GameState::InProgress {
                                    let ai_move = if current_player == a1_symbol {
                                        ai_1.symbol = current_player;
                                        ai_1.get_move(&board)
                                    } else {
                                        ai_2.symbol = current_player;
                                        ai_2.get_move(&board)
                                    };

                                    board.make_move(ai_move, current_player);

                                    current_player = current_player.opponent();
                                }

                                match board.check_winner() {
                                    GameState::XWins => {
                                        if a1_symbol == Cell::X {
                                            ai_1_wins += 1;
                                        } else {
                                            ai_2_wins += 1;
                                        }
                                    }
                                    GameState::OWins => {
                                        if a1_symbol == Cell::O {
                                            ai_1_wins += 1;
                                        } else {
                                            ai_2_wins += 1;
                                        }
                                    }
                                    GameState::Draw => {
                                        draws += 1;
                                    }
                                    _ => {}
                                }
                            }

                            println!("\nBatch {} results:", batch);
                            println!("  AI-1 wins: {}", ai_1_wins);
                            println!("  AI-2 wins: {}", ai_2_wins);
                            println!("  Draws: {}", draws);

                            if ai_1_wins > ai_2_wins {
                                println!("  {}AI-1 performed better!{}", GREEN, RESET);
                                self.ai.nn = ai_1.nn.clone();
                                ai_2.nn = ai_1.nn.clone();
                            } else if ai_2_wins > ai_1_wins {
                                println!("  {}AI-2 performed better!{}", GREEN, RESET);
                                self.ai.nn = ai_2.nn.clone();
                                ai_1.nn = ai_2.nn.clone();
                            } else {
                                println!("  {}It's a tie! No clear winner.{}", YELLOW, RESET);
                            }

                            if let Err(e) = self.ai.nn.save_to_file(MODEL_FILE) {
                                eprintln!("Failed to save model: {}", e);
                            } else {
                                println!("  Saved best model to {}", MODEL_FILE);
                            }

                            let mut rng = rand::rng();

                            if ai_1_wins <= ai_2_wins {
                                println!("  Training AI-1 for next batch...");
                                let mut board = Board::new();
                                let mut game_history: Vec<(Board, usize)> = Vec::new();
                                let mut current_player = Cell::X;

                                while board.check_winner() == GameState::InProgress {
                                    let board_before = board.clone();
                                    let empty_cells = board.get_empty_cells();
                                    let random_move = *empty_cells.choose(&mut rng).unwrap();

                                    board.make_move(random_move, current_player);
                                    game_history.push((board_before, random_move));
                                    current_player = current_player.opponent();
                                }

                                let result = board.check_winner();

                                ai_1.symbol = Cell::X;
                                ai_1.record_game(&game_history, result);
                                ai_1.train_incremental(3);

                                ai_1.symbol = Cell::O;
                                ai_1.record_game(&game_history, result);
                                ai_1.train_incremental(3);
                            }

                            if ai_2_wins <= ai_1_wins {
                                println!("  Training AI-2 for next batch...");
                                let mut board = Board::new();
                                let mut game_history: Vec<(Board, usize)> = Vec::new();
                                let mut current_player = Cell::X;

                                while board.check_winner() == GameState::InProgress {
                                    let board_before = board.clone();
                                    let empty_cells = board.get_empty_cells();
                                    let random_move = *empty_cells.choose(&mut rng).unwrap();

                                    board.make_move(random_move, current_player);
                                    game_history.push((board_before, random_move));
                                    current_player = current_player.opponent();
                                }

                                let result = board.check_winner();

                                ai_2.symbol = Cell::X;
                                ai_2.record_game(&game_history, result);
                                ai_2.train_incremental(3);

                                ai_2.symbol = Cell::O;
                                ai_2.record_game(&game_history, result);
                                ai_2.train_incremental(3);
                            }
                        }

                        self.ai.symbol = self.ai_symbol;

                        println!("\n{}Competitive training complete!{}", GREEN, RESET);
                        println!("The neural network has improved through competition.");
                        println!("\nPress Enter to continue...");
                        io::stdout().flush().unwrap();
                        let mut input = String::new();
                        io::stdin().read_line(&mut input).unwrap();
                    }
                } else {
                    println!("{}Invalid input. Training cancelled.{}", RED, RESET);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        } else {
            println!("{}Invalid input. Training cancelled.{}", RED, RESET);
            thread::sleep(Duration::from_secs(1));
        }
    }

    fn watch_ai_vs_ai(&mut self) {
        clear_screen();
        println!("{}{}WATCH AI VS AI GAME{}{}", BOLD, CYAN, RESET, YELLOW);
        println!("-------------------");
        println!("Watch the Neural Network play against itself!");
        println!("Press Enter after each move to continue, or 'q' to quit anytime.");
        println!();

        let mut board = Board::new();
        let mut game_history: Vec<(Board, usize)> = Vec::new();
        let mut current_player = Cell::X;

        let mut ai_x = NeuralNetworkAI::new(Cell::X);
        let mut ai_o = NeuralNetworkAI::new(Cell::O);

        ai_x.nn = self.ai.nn.clone();
        ai_o.nn = self.ai.nn.clone();

        display_spectator_board(&board, current_player);

        println!("\nPress Enter to start the game...");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let mut move_count = 0;

        while board.check_winner() == GameState::InProgress {
            move_count += 1;

            let board_before = board.clone();

            let ai_move = if current_player == Cell::X {
                ai_x.get_move(&board)
            } else {
                ai_o.get_move(&board)
            };

            board.make_move(ai_move, current_player);

            game_history.push((board_before, ai_move));

            display_spectator_board(&board, current_player.opponent());

            println!(
                "Move #{}: {} placed at position {}",
                move_count,
                if current_player == Cell::X {
                    "X (Green)"
                } else {
                    "O (Red)"
                },
                ai_move + 1
            );

            let state = board.check_winner();
            if state != GameState::InProgress {
                println!("\n{}GAME OVER{}", BOLD, RESET);
                match state {
                    GameState::XWins => println!("{}X (Green) wins!{}", GREEN, RESET),
                    GameState::OWins => println!("{}O (Red) wins!{}", RED, RESET),
                    GameState::Draw => println!("{}It's a draw!{}", YELLOW, RESET),
                    _ => {}
                }

                println!("\n{}AI is learning from this game...{}", PURPLE, RESET);
                io::stdout().flush().unwrap();

                self.ai.symbol = Cell::X;
                self.ai.record_game(&game_history, state);
                self.ai.train_incremental(5);

                self.ai.symbol = Cell::O;
                self.ai.record_game(&game_history, state);
                self.ai.train_incremental(5);

                self.ai.symbol = self.ai_symbol;

                println!("{}Learning complete!{}", GREEN, RESET);

                if let Err(e) = self.ai.nn.save_to_file(MODEL_FILE) {
                    eprintln!("Failed to save model: {}", e);
                }

                println!("\nPress Enter to return to the main menu...");
                io::stdout().flush().unwrap();
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                break;
            }

            current_player = current_player.opponent();

            thread::sleep(Duration::from_secs(1));
        }
    }

    fn display_stats(&self) {
        clear_screen();
        println!("{}{}GAME STATISTICS{}{}", BOLD, BLUE, RESET, YELLOW);
        println!("----------------");
        println!("Total games played: {}", self.games_played);
        println!(
            "Player wins: {} ({:.1}%)",
            self.player_wins,
            if self.games_played > 0 {
                (self.player_wins as f32 / self.games_played as f32) * 100.0
            } else {
                0.0
            }
        );
        println!(
            "AI wins: {} ({:.1}%)",
            self.ai_wins,
            if self.games_played > 0 {
                (self.ai_wins as f32 / self.games_played as f32) * 100.0
            } else {
                0.0
            }
        );
        println!(
            "Draws: {} ({:.1}%)",
            self.draws,
            if self.games_played > 0 {
                (self.draws as f32 / self.games_played as f32) * 100.0
            } else {
                0.0
            }
        );

        if let Ok(metadata) = fs::metadata(MODEL_FILE) {
            println!("\nNeural Network Model:");
            println!("  File: {}", MODEL_FILE);
            println!("  Size: {} bytes", metadata.len());
            println!("  Training examples: {}", self.ai.training_data.len());
        } else {
            println!("\nNeural Network Model: Not yet created");
        }

        println!();
        println!(
            "AI Performance: {}",
            if self.games_played < 5 {
                "Not enough data"
            } else if self.ai_wins as f32 / self.games_played as f32 > 0.6 {
                "Strong"
            } else if self.ai_wins as f32 / self.games_played as f32 > 0.4 {
                "Moderate"
            } else {
                "Needs more training"
            }
        );

        println!();
        println!("Press Enter to return to the main menu...");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
    }

    fn run(&mut self) {
        loop {
            self.display_main_menu();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            match input {
                "1" => {
                    self.player_symbol = Cell::X;
                    self.ai_symbol = Cell::O;
                    self.current_turn = Player::Human;

                    if let Ok(model) = NeuralNetwork::from_file(MODEL_O_FILE) {
                        println!("{}Using specialized O model for AI{}", PURPLE, RESET);
                        self.ai.nn = model;
                        self.ai.symbol = Cell::O;
                        thread::sleep(Duration::from_millis(1000));
                    }

                    self.play_game();
                }
                "2" => {
                    self.player_symbol = Cell::O;
                    self.ai_symbol = Cell::X;
                    self.current_turn = Player::AI;

                    if let Ok(model) = NeuralNetwork::from_file(MODEL_X_FILE) {
                        println!("{}Using specialized X model for AI{}", PURPLE, RESET);
                        self.ai.nn = model;
                        self.ai.symbol = Cell::X;
                        thread::sleep(Duration::from_millis(1000));
                    }

                    self.play_game();
                }
                "3" => {
                    self.train_ai();
                }
                "4" => {
                    self.train_specialized_ai();
                }
                "5" => {
                    self.watch_ai_vs_ai();
                }
                "6" => {
                    self.display_stats();
                }
                "7" => {
                    clear_screen();
                    println!("{}Thanks for playing!{}", GREEN, RESET);
                    break;
                }
                _ => {
                    clear_screen();
                    println!("{}Invalid choice. Please try again.{}", RED, RESET);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        }
    }
}
fn display_spectator_board(board: &Board, current_player: Cell) {
    clear_screen();
    println!("{}{}WATCHING AI VS AI GAME{}{}", BOLD, CYAN, RESET, YELLOW);
    println!("----------------------");
    println!("{}X (Green){} vs {}O (Red){}", GREEN, RESET, RED, RESET);

    println!(
        "\nCurrent turn: {}",
        if current_player == Cell::X {
            format!("{}X's TURN{}", GREEN, RESET)
        } else {
            format!("{}O's TURN{}", RED, RESET)
        }
    );

    println!("\n");
    println!(
        " {} | {} | {} ",
        board.cells[0].to_colored_string(),
        board.cells[1].to_colored_string(),
        board.cells[2].to_colored_string()
    );
    println!("-----------");
    println!(
        " {} | {} | {} ",
        board.cells[3].to_colored_string(),
        board.cells[4].to_colored_string(),
        board.cells[5].to_colored_string()
    );
    println!("-----------");
    println!(
        " {} | {} | {} ",
        board.cells[6].to_colored_string(),
        board.cells[7].to_colored_string(),
        board.cells[8].to_colored_string()
    );
    println!();

    println!("{}Board positions:{}", CYAN, RESET);
    println!(" 1 | 2 | 3 ");
    println!("-----------");
    println!(" 4 | 5 | 6 ");
    println!("-----------");
    println!(" 7 | 8 | 9 ");
    println!();
}

fn clear_screen() {
    print!("{}", CLEAR_SCREEN);
    io::stdout().flush().unwrap();
}

fn main() {
    let mut game = TicTacToe::new();
    game.run();
}
