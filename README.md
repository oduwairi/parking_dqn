# Autonomous Parking with Deep Q-Learning (DQN)

**Research Implementation**: Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach

## Project Overview

This project implements a Deep Q-Learning approach for autonomous parking in a 2D simulated environment with static and dynamic obstacles. The agent learns to navigate and park successfully using reinforcement learning techniques.

## Phase-by-Phase Implementation Roadmap

### Phase 1: Environment Setup & Basic Simulation 🏗️
**Duration**: 3-5 days  
**Status**: 🚧 IN PROGRESS

**Objectives**:
- Set up the OpenAI Gym-compatible simulation environment
- Implement 2D parking lot layout with basic rendering
- Create the car agent with kinematic motion model
- Implement basic state space representation

**Key Components**:
- [x] Project structure and dependencies (`requirements.txt`)
- [x] Basic 2D parking environment class
- [x] Car agent with position, orientation, velocity
- [x] Kinematic motion equations implementation:
  - `x_{t+1} = x_t + v_t * cos(θ_t) * Δt`
  - `y_{t+1} = y_t + v_t * sin(θ_t) * Δt` 
  - `θ_{t+1} = θ_t + (v_t/L) * tan(δ_t) * Δt`
- [x] Basic visualization and rendering
- [x] Simple state vector: `[x_t, y_t, θ_t, v_t, d_1...d_8]`

**Files Created**:
- [x] `src/environment/parking_env.py` - Main environment class
- [x] `src/environment/car_agent.py` - Car physics and kinematics
- [x] `src/environment/renderer.py` - Visualization components
- [x] `requirements.txt` - Project dependencies
- [x] `test_phase1.py` - Phase 1 testing script

**Next Steps**:
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Test Phase 1**: `python test_phase1.py`
3. **Verify**: Ensure all tests pass and pygame window displays correctly

---

### Phase 2: Action Space & Basic Interaction 🎮
**Duration**: 2-3 days  
**Status**: ✅ COMPLETE

**Objectives**:
- Implement discrete action space (7 actions as per paper)
- Add distance sensors for obstacle detection
- Create basic reward function structure
- Test agent-environment interaction loop

**Key Components**:
- [x] Action space implementation with 7 discrete actions:
  - `a_0`: Hold/Brake (Δv = -0.6 m/s, Δδ = 0°)
  - `a_1`: Throttle forward (Δv = +0.6 m/s, Δδ = 0°)
  - `a_2`: Reverse back (Δv = -0.6 m/s, Δδ = 0°)
  - `a_3`: Left forward (Δv = +0.6 m/s, Δδ = +8°)
  - `a_4`: Right forward (Δv = +0.6 m/s, Δδ = -8°)
  - `a_5`: Left reverse (Δv = -0.6 m/s, Δδ = +8°)
  - `a_6`: Right reverse (Δv = -0.6 m/s, Δδ = -8°)
- [x] 8-directional distance sensors (ultrasonic/LiDAR simulation)
- [x] Basic reward function framework
- [x] Episode termination conditions

**Files Created**:
- [x] `src/environment/action_space.py` - Action definitions and logic
- [x] `src/environment/sensors.py` - Distance sensor implementations
- [x] `src/environment/rewards.py` - Reward function calculations
- [x] `test_phase2.py` - Phase 2 testing script

**Test Results**:
1. **Action Space Tests**: ✅ PASSED - All 7 actions implemented per paper specifications
2. **Sensor System Tests**: ✅ PASSED - 8-directional sensors with boundary detection
3. **Reward Function Tests**: ✅ PASSED - Collision (-100), Success (+100), Progress (±1/-0.5), Time (-0.1)
4. **Environment Integration**: ✅ PASSED - Full integration with modular components
5. **Interactive Demo**: ✅ PASSED - Real-time visualization with sensor rays

**Next Steps**:
1. **Completed**: All Phase 2 objectives achieved
2. **Test Phase 2**: `python test_phase2.py` ✅ ALL TESTS PASS
3. **Begin Phase 3**: Static Obstacles & Reward Engineering

---

### Phase 3: Static Obstacles & Reward Engineering 🚧
**Duration**: 3-4 days  
**Status**: 📋 Planned

**Objectives**:
- Add static obstacles (barriers, stationary vehicles)
- Implement comprehensive reward function
- Create parking spot detection and validation
- Test collision detection system

**Key Components**:
- [ ] Static obstacle placement and collision detection
- [ ] Comprehensive reward function:
  - Collision penalty: -100 (episode termination)
  - Success reward: +100 (episode termination) 
  - Progress reward: +1 (closer to target), -0.5 (further)
  - Time penalty: -0.1 (per timestep)
- [ ] Parking spot validation (position tolerance ε_p = 0.5m, orientation tolerance ε_θ = 10°)
- [ ] Episode management and reset functionality

**Files to Create**:
- `src/environment/obstacles.py` - Static obstacle management
- `src/environment/parking_spots.py` - Parking validation logic
- `src/environment/collision_detection.py` - Collision detection system

---

### Phase 4: DQN Network Architecture 🧠
**Duration**: 4-5 days  
**Status**: ✅ COMPLETE

**Objectives**:
- Implement main and target DQN networks
- Create neural network architecture (3 hidden layers, 256 neurons each)
- Implement experience replay buffer
- Set up training infrastructure

**Key Components**:
- [x] DQN network architecture:
  - Input: State vector [x, y, θ, v, d_1...d_8] (12 dimensions)
  - Hidden layers: 3 layers × 256 neurons + ReLU activation
  - Output: Q-values for 7 actions
- [x] Main network and target network (θ and θ⁻)
- [x] Experience replay buffer (capacity ~10⁵ transitions)
- [x] Huber loss function implementation
- [x] Epsilon-greedy policy with exponential decay
- [x] Double DQN implementation (reduces overestimation bias)
- [x] Prioritized experience replay (optional)

**Files Created**:
- [x] `src/dqn/network.py` - DQN network architecture
- [x] `src/dqn/replay_buffer.py` - Experience replay implementation
- [x] `src/dqn/agent.py` - DQN agent with epsilon-greedy policy
- [x] `src/dqn/loss_functions.py` - Huber loss and training utilities
- [x] `test_phase4.py` - Phase 4 testing script

**Test Results**:
1. **DQN Network Architecture**: ✅ PASSED - 3×256 neural network with 136,711 parameters
2. **Experience Replay Buffer**: ✅ PASSED - Efficient circular buffer with 1MB memory usage
3. **Loss Functions & Utilities**: ✅ PASSED - Huber loss, epsilon/LR scheduling
4. **DQN Agent Integration**: ✅ PASSED - Complete agent with save/load functionality
5. **Environment Integration**: ✅ PASSED - Full integration with parking environment
6. **Performance Benchmarks**: ✅ PASSED - 207 training steps/sec, 4,395 actions/sec

**Next Steps**:
1. **Completed**: All Phase 4 objectives achieved with excellent performance
2. **Test Phase 4**: `python test_phase4.py` ✅ ALL TESTS PASS
3. **Begin Phase 5**: Training Pipeline & Hyperparameter Setup

---

### Phase 5: Training Pipeline & Hyperparameter Setup ⚙️
**Duration**: 3-4 days  
**Status**: ✅ COMPLETE

**Objectives**:
- Implement complete training loop
- Set up hyperparameter configuration
- Add training monitoring and logging
- Implement target network soft updates

**Key Components**:
- [x] Training hyperparameters (as per paper):
  - Learning rate α = 10⁻³
  - Discount factor γ = 0.9-0.95
  - Batch size B = 64
  - Target update frequency N = 1000
  - Soft update rate τ = 10⁻³
- [x] Epsilon decay: `ε_t = ε_min + (ε_max - ε_min) * exp(-λt)`
- [x] Training loop with gradient descent and backpropagation
- [x] Model checkpointing and saving
- [x] Training metrics logging (rewards, losses, epsilon values)

**Files to Create**:
- `src/training/trainer.py` - Main training loop
- `src/training/config.py` - Hyperparameter configuration
- `src/training/logger.py` - Training metrics and logging
- `src/training/checkpoint.py` - Model saving/loading utilities

---

### Phase 6: Dynamic Obstacles & Advanced Scenarios 🚶‍♂️🚗
**Duration**: 4-5 days  
**Status**: 📋 Planned

**Objectives**:
- Add dynamic obstacles (moving pedestrians, cars)
- Create diverse training scenarios
- Implement advanced parking configurations
- Test robustness with varying conditions

**Key Components**:
- [ ] Dynamic obstacle system (pedestrians, moving vehicles)
- [ ] Multiple parking lot configurations
- [ ] Random scenario generation for training diversity
- [ ] Advanced collision avoidance scenarios
- [ ] Varying parking spot sizes and orientations

**Files to Create**:
- `src/environment/dynamic_obstacles.py` - Moving obstacle system
- `src/environment/scenario_generator.py` - Random scenario creation
- `src/environment/advanced_layouts.py` - Complex parking configurations

---

### Phase 7: Training Execution & Optimization 🏃‍♂️
**Duration**: 5-7 days  
**Status**: ✅ COMPLETE

**Objectives**:
- Execute full training (5000 episodes)
- Monitor convergence and performance
- Optimize hyperparameters if needed
- Generate training analytics and curves

**Key Components**:
- [x] Progressive training system with 3 stages:
  - Stage 1: Simple environment (no obstacles, 1000 episodes)
  - Stage 2: Add obstacles (1500 episodes)
  - Stage 3: Full complexity with random targets (2000 episodes)
- [x] Transfer learning between stages (models build on previous stage)
- [x] GPU acceleration support with automatic fallback to CPU
- [x] Real-time visualization during training (always-on pygame window)
- [x] Comprehensive training monitoring and logging
- [x] Automatic progression criteria and success/failure detection
- [x] Training curve generation and performance analytics
- [x] Early stopping and convergence detection
- [x] Model checkpointing and best model selection

**Files Created**:
- [x] Enhanced `src/training/train_with_viz.py` - Progressive training with visualization
- [x] Extended `src/training/config.py` - Progressive stage configurations
- [x] Enhanced `src/training/trainer.py` - GPU support and comprehensive logging

**Training Configurations**:
- [x] `progressive_simple`: 1000 episodes, no obstacles, fixed target
- [x] `progressive_obstacles`: 1500 episodes, with obstacles, transfer learning
- [x] `progressive_full`: 2000 episodes, full complexity, transfer learning
- [x] GPU-optimized settings with large batch sizes and efficient memory usage

**Usage Examples**:
```bash
# Full progressive training (3 stages with transfer learning)
python src/training/train_with_viz.py --mode progressive

# Single stage training with visualization
python src/training/train_with_viz.py --mode single --stage progressive_simple

# Start from specific stage (e.g., stage 2 if stage 1 already complete)
python src/training/train_with_viz.py --mode progressive --start-stage 1
```

**Key Features Implemented**:
- 🎮 **Always-on visualization**: Pygame window shows agent behavior every episode
- 🔄 **Transfer learning**: Each stage builds on previous stage's learned weights
- 📊 **Comprehensive logging**: Real-time metrics, training curves, performance tracking
- 🎯 **Automatic progression**: Stages advance based on success criteria (e.g., ≥20% success for Stage 1)
- 🚀 **GPU acceleration**: Automatic CUDA detection with CPU fallback
- 📈 **Training analytics**: Loss curves, reward progression, success/collision rates
- 💾 **Smart checkpointing**: Best models saved, training resumable from any point

**Next Steps**:
1. **Completed**: Progressive training system fully operational
2. **Current**: GPU PyTorch installation for acceleration
3. **Begin Phase 8**: Evaluation & Testing Framework

---

### Phase 8: Evaluation & Testing Framework 📊
**Duration**: 3-4 days  
**Status**: 📋 Planned

**Objectives**:
- Implement comprehensive evaluation metrics
- Create test scenarios separate from training
- Measure success rates, collision rates, timing
- Generate performance reports

**Key Components**:
- [ ] Evaluation metrics implementation:
  - Success rate (target: ≥70%)
  - Average time to park
  - Collision count (target: ≤1%)
  - Parking accuracy (position and orientation errors)
- [ ] Test scenario generation (different from training)
- [ ] Statistical analysis and confidence intervals
- [ ] Performance comparison with baseline methods

**Files to Create**:
- `src/evaluation/metrics.py` - Evaluation metrics calculation
- `src/evaluation/test_scenarios.py` - Test case generation
- `src/evaluation/evaluator.py` - Main evaluation pipeline
- `src/evaluation/report_generator.py` - Performance reporting

---

### Phase 9: Visualization & Analysis Tools 📈
**Duration**: 2-3 days  
**Status**: 📋 Planned

**Objectives**:
- Create visualization tools for agent behavior
- Implement trajectory plotting and analysis
- Generate research-quality plots and charts
- Create demo videos of successful parking

**Key Components**:
- [ ] Agent trajectory visualization
- [ ] Parking success/failure analysis plots
- [ ] Q-value heatmaps and decision analysis
- [ ] Training progress visualization
- [ ] Demo video generation capability

**Files to Create**:
- `src/visualization/trajectory_plotter.py` - Path visualization
- `src/visualization/performance_plots.py` - Metrics visualization
- `src/visualization/demo_generator.py` - Video/GIF creation

---

### Phase 10: Documentation & Final Integration 📚
**Duration**: 2-3 days  
**Status**: 📋 Planned

**Objectives**:
- Complete documentation and user guides
- Final code cleanup and optimization
- Integration testing and bug fixes
- Prepare research artifacts and reproducibility package

**Key Components**:
- [ ] Complete API documentation
- [ ] User guide and setup instructions
- [ ] Reproducibility package with trained models
- [ ] Final performance validation
- [ ] Code quality improvements and refactoring

---

## Project Structure

```
parking_dqn/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── setup.py                          # Package setup
├── test_phase1.py                    # Phase 1 testing script
├── test_phase2.py                    # Phase 2 testing script
├── test_phase3.py                    # Phase 3 testing script
├── test_phase4.py                    # Phase 4 testing script ✅ NEW
├── config/                           # Configuration files
│   ├── training_config.yaml
│   ├── environment_config.yaml
│   └── evaluation_config.yaml
├── src/                              # Source code
│   ├── __init__.py                   # ✅ Created
│   ├── environment/                  # Simulation environment
│   │   ├── __init__.py               # ✅ Created
│   │   ├── parking_env.py           # ✅ Created - Main environment
│   │   ├── car_agent.py             # ✅ Created - Car physics
│   │   ├── renderer.py              # ✅ Created - Visualization
│   │   ├── action_space.py          # ✅ Created - Action definitions
│   │   ├── sensors.py               # ✅ Created - Distance sensors
│   │   ├── obstacles.py             # ✅ Created - Static obstacles
│   │   ├── parking_spots.py         # ✅ Created - Parking validation
│   │   ├── collision_detection.py   # ✅ Created - Collision system
│   │   ├── rewards.py               # ✅ Created - Reward functions
│   │   ├── dynamic_obstacles.py     # Moving obstacles
│   │   ├── scenario_generator.py    # Random scenarios
│   │   └── advanced_layouts.py      # Complex layouts
│   ├── dqn/                         # Deep Q-Network
│   │   ├── __init__.py               # ✅ Created
│   │   ├── network.py               # ✅ Created - DQN architecture
│   │   ├── agent.py                 # ✅ Created - DQN agent
│   │   ├── replay_buffer.py         # ✅ Created - Experience replay
│   │   └── loss_functions.py        # ✅ Created - Loss functions
│   ├── training/                    # Training pipeline
│   │   ├── __init__.py               # ✅ Created
│   │   ├── trainer.py               # Main training loop
│   │   ├── train_main.py            # Training script
│   │   ├── config.py                # Hyperparameters
│   │   ├── logger.py                # Logging utilities
│   │   └── checkpoint.py            # Model management
│   ├── evaluation/                  # Testing & evaluation
│   │   ├── __init__.py               # ✅ Created
│   │   ├── evaluator.py             # Main evaluation
│   │   ├── metrics.py               # Performance metrics
│   │   ├── test_scenarios.py        # Test cases
│   │   └── report_generator.py      # Reporting
│   ├── visualization/               # Visualization tools
│   │   ├── __init__.py               # ✅ Created
│   │   ├── trajectory_plotter.py    # Path plotting
│   │   ├── performance_plots.py     # Metrics plots
│   │   └── demo_generator.py        # Video generation
│   └── analysis/                    # Analysis tools
│       ├── __init__.py               # ✅ Created
│       ├── training_monitor.py      # Training monitoring
│       └── plot_results.py          # Result plotting
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_dqn.py
│   ├── test_training.py
│   └── test_evaluation.py
├── models/                          # Saved models
│   └── checkpoints/                 # ✅ Created
├── data/                            # Training data
│   ├── training_logs/               # ✅ Created
│   └── evaluation_results/          # ✅ Created
├── notebooks/                       # Jupyter notebooks
│   ├── exploration.ipynb
│   ├── analysis.ipynb
│   └── visualization.ipynb
└── docs/                           # Documentation
    ├── api_reference.md
    ├── user_guide.md
    └── research_notes.md
```

## Key Research Parameters (From Paper)

### State Space
- Position: (x, y) coordinates
- Orientation: θ (angle)
- Velocity: v
- Distance sensors: d₁...d₈ (8 directional sensors)
- **Vector**: `s_t = [x_t, y_t, θ_t, v_t, d_1...d_8]ᵀ`

### Action Space (7 discrete actions)
| ID | Action | Description | Δδ (steer) | Δv (m/s) |
|----|--------|-------------|------------|----------|
| 0  | a₀     | Hold (brake) | 0° | -0.6 |
| 1  | a₁     | Throttle forward | 0° | +0.6 |
| 2  | a₂     | Reverse back | 0° | -0.6 |
| 3  | a₃     | Left forward | +8° | +0.6 |
| 4  | a₄     | Right forward | -8° | +0.6 |
| 5  | a₅     | Left reverse | +8° | -0.6 |
| 6  | a₆     | Right reverse | -8° | -0.6 |

### Reward Function
- **Collision Penalty**: -100 (episode termination)
- **Success Reward**: +100 (episode termination)
- **Progress Reward**: +1 (closer to target), -0.5 (further)
- **Time Penalty**: -0.1 (per timestep)
- **Tolerances**: Position ε_p = 0.5m, Orientation ε_θ = 10°

### Hyperparameters
- **Learning Rate**: α = 10⁻³
- **Discount Factor**: γ = 0.9-0.95
- **Batch Size**: B = 64
- **Replay Buffer**: ~10⁵ transitions
- **Target Update**: Every N = 1000 steps
- **Soft Update Rate**: τ = 10⁻³
- **Training Episodes**: 5000
- **Network Architecture**: 3 hidden layers × 256 neurons + ReLU

### Success Criteria
- **Success Rate**: ≥70%
- **Collision Rate**: ≤1%
- **Training Convergence**: Stable improvement over episodes

## Getting Started

### 1. Clone and Setup
```bash
git clone <repository-url>
cd parking_dqn
pip install -r requirements.txt
```

### 2. Test Phase 1 Installation
```bash
python test_phase1.py
```

This will run a comprehensive test of the Phase 1 implementation including:
- ✅ Environment setup verification
- ✅ Car physics and action testing
- ✅ Reward system validation
- ✅ Interactive visualization demo

### 3. Expected Output
If Phase 1 is working correctly, you should see:
- Console output showing all tests passing
- A pygame window displaying the parking environment
- A blue car (agent) and green parking spot (target)
- Real-time position, orientation, and sensor data

### 4. Next Steps
Once Phase 1 tests pass:
1. **Document any issues** in `docs/research_notes.md`
2. **Update this README** with Phase 1 completion status
3. **Begin Phase 2** implementation (action space & sensors)

## Phase 1 Troubleshooting

### Common Issues:
1. **Pygame not displaying**: Ensure you have a display available
2. **Import errors**: Verify all dependencies installed correctly
3. **Performance issues**: Check that numpy/torch are properly installed

### Debug Commands:
```bash
# Test basic imports
python -c "import src.environment.parking_env; print('✅ Imports working')"

# Test without visualization
python -c "from src.environment.parking_env import ParkingEnv; env = ParkingEnv(); print('✅ Environment created')"
```

## Research Notes

This implementation follows the methodology described in "Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach" by Osama ALDuwairi. The project aims to validate the proposed DQN approach for autonomous parking with measurable success criteria.

---

**Current Status**: Phase 4 implementation complete and tested. DQN network architecture with 136,711 parameters fully operational and integrated with autonomous parking environment. Next step is to implement the complete training pipeline (Phase 5). 