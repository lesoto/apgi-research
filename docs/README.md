# APGI Research Application - Program Overview

## Introduction

The APGI Research Application is an autonomous research system designed to optimize psychological experiments through AI-driven parameter tuning and continuous learning.

## Architecture

The APGI Research Application implements a complete autonomous learning algorithm (ALGORITH) with human-in-the-loop oversight:

### Core Components

1. **Human Control Layer** (`human_layer.py`)
   - `HumanControlLayer` - Interactive configuration and approval workflow
   - Task prioritization with `TaskPriority` enum (CRITICAL, HIGH, MEDIUM, LOW)
   - Review & Decision system with `ReviewDecision` enum (APPROVE, MODIFY, REJECT)
   - Three interaction modes: interactive, batch, autonomous
   - Integration with `hypothesis_approval_board.py` for hypothesis management

2. **Agent Harness** (`autonomous_agent.py`)
   - `AutonomousAgent` - Main controller orchestrating optimization loops
   - `GitPerformanceTracker` - Version control-based performance tracking with async operations
   - `ParameterOptimizer` - AI-driven parameter modification with experiment-specific strategies
   - `AsyncGitOperations` - Non-blocking git commits for continuous optimization
   - Checkpoint system for crash recovery

3. **Agent Execution Engine** (`xpr_agent_engine.py`)
   - `XPRAgentEngine` - Core engine with skill registration and execution
   - `XPRAgentEngineEnhanced` - Advanced engine with LLM integration via litellm
   - `LLMIntegration` - Multi-provider LLM support (OpenAI, Anthropic, local models)
   - Self-healing skill chain: `xpr_job_debug` → `xpr_issue_fix` → `xpr_issue_report`
   - LLM-generated code patches with syntax validation

4. **Memory System** (`memory_store.py`)
   - `MemoryStore` - Indexed knowledge base with TF-IDF semantic search
   - `MemoryEntry` - Structured storage for patterns, strategies, and failures
   - `VectorEmbedding` - Neural-style embeddings for semantic similarity
   - Hybrid search combining TF-IDF and vector similarity
   - `update_memory_from_report()` - Automatic extraction of lessons learned

5. **Experiment GUI** (`GUI-auto_improve_experiments.py`)
   - `ExperimentRunnerGUI` - Modern CustomTkinter interface
   - Real-time experiment execution and visualization
   - Hypothesis approval board integration
   - Results visualization with matplotlib

6. **APGI Core** (`APGI_System.py`)
   - Complete dynamical system implementation
   - 51 psychological states with Π vs Π̂ distinction
   - Neuromodulator mapping (ACh, NE, DA, 5-HT)
   - Measurement equations (HEP, P3b, detection thresholds)

### Key Features

- **Human-in-the-Loop Control**: Interactive approval workflow with APPROVE/MODIFY/REJECT decisions
- **Self-Healing Experiments**: Automatic error detection, debugging, and retry with skill chaining
- **Skill Chaining System**: Modular agent skills (debug → fix → report) for autonomous recovery
- **Cognitive Memory**: TF-IDF and vector embeddings for pattern recognition across iterations
- **Infinite Autonomous Optimization**: Continuous parameter optimization with async git operations
- **Comprehensive Guardrails**: Confidence thresholds, metric regression detection, safety violations
- **Git-Based Learning**: Version control tracking with automatic rollback on failures
- **Multi-Experiment Support**: 30+ psychological experiments with experiment-specific strategies
- **LLM Integration**: Multi-provider support (OpenAI, Anthropic, local) via litellm
- **Real-Time Monitoring**: GUI with live experiment execution and embedded matplotlib visualization
- **Crash Recovery**: Checkpoint system for resuming after interruptions
- **Security Hardened**: Parameter whitelist, module validation, and sandboxed execution

## Supported Experiments

The application supports a comprehensive suite of psychological experiments:

- **Attentional Blink** - Temporal attention dynamics
- **Iowa Gambling Task** - Decision making under uncertainty
- **Stroop Effect** - Cognitive interference and control
- **Visual Search** - Feature-based and conjunction search
- **Change Blindness** - Visual perception and attention
- **Masking** - Visual temporal processing
- **Binocular Rivalry** - Visual perception competition
- **Posner Cueing** - Spatial attention orienting
- **Simon Effect** - Stimulus-response compatibility
- **Flanker Task** - Selective attention and inhibition
- **Stop Signal** - Response inhibition and control
- **Dual N-Back** - Working memory capacity
- **Working Memory Span** - Short-term memory limits
- **Time Estimation** - Temporal perception accuracy
- **Probabilistic Category Learning** - Learning under uncertainty
- **Artificial Grammar Learning** - Implicit learning patterns

## Usage

### Command Line Interface

```bash
# Optimize a single experiment
python autonomous_agent.py --experiment masking --iterations 100

# Optimize all experiments
python autonomous_agent.py --all-experiments --iterations 50

# Run overnight optimization (8 hours)
python autonomous_agent.py --overnight

# Run with specific timeout
python autonomous_agent.py --experiment stroop_effect --timeout 3600
```

### GUI Interface

```bash
# Launch the graphical interface
python GUI-auto_improve_experiments.py
```

The GUI provides:

- Real-time experiment execution
- Visual performance monitoring
- Sequential and parallel run modes
- Dependency management
- Results visualization

### API Integration

```python
from autonomous_agent import AutonomousAgent
from human_layer import HumanControlLayer, ReviewDecision
from memory_store import MemoryStore
from xpr_agent_engine import XPRAgentEngineEnhanced

# Initialize components
agent = AutonomousAgent()
human = HumanControlLayer()
memory = MemoryStore()
engine = XPRAgentEngineEnhanced()

# Configure human interaction layer
human.configure_if_needed()  # Interactive setup wizard

# Select and prioritize tasks
task = human.select_task()

# Run optimization with human oversight
results = agent.optimize_experiment("attentional_blink", iterations=100)

# Run single experiment with modifications
result = agent.run_experiment(
    "masking",
    modifications={"BASE_DETECTION_RATE": 0.5},
    timeout_seconds=1800,
    max_retries=1  # Self-healing retry on failure
)

# Review results with human decision
review_result = human.review({
    "experiment_id": result.experiment_name,
    "metrics": {"accuracy": result.primary_metric},
    "confidence": 0.85
})

# Update memory with lessons learned
if review_result.decision == ReviewDecision.APPROVE:
    memory.add_memory_with_embedding(
        experiment_name="masking",
        pattern_type="success_pattern",
        content="BASE_DETECTION_RATE=0.5 improved performance by 15%",
        context={"metric_delta": 0.15}
    )
```

## Configuration

### Environment Setup

1. **Python Requirements** (Python 3.8+)

   ```bash
   pip install -r requirements.txt
   ```

2. **Optional Dependencies**

   ```bash
   pip install -r requirements-optional.txt
   ```

3. **Development Dependencies**

   ```bash
   pip install -r requirements-test.txt
   ```

### Experiment Configuration

Each experiment is configured through Python files:

- `prepare_<experiment>.py` - Setup and parameter definitions
- `run_<experiment>.py` - Main experiment execution

### Performance Tuning

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Memory Optimization**: Efficient data structures and caching
- **Parallel Processing**: Multi-threaded experiment execution
- **Timeout Management**: Configurable experiment timeouts

## Security Features

### Input Validation

- Parameter whitelist enforcement
- Module name validation
- File path sanitization
- Package name validation

### Execution Security

- Sandboxed subprocess execution
- Git staging restrictions
- Module cache invalidation
- Secure import validation

### Data Protection

- No sensitive data in git commits
- Encrypted parameter storage
- Audit logging
- Error boundary enforcement

## Performance Metrics

### Optimization Algorithms

- **Genetic Algorithms**: Population-based optimization
- **Bayesian Optimization**: Model-based search
- **Random Search**: Baseline comparison
- **Gradient Methods**: When applicable

### Evaluation Metrics

- **Primary Metrics**: Experiment-specific performance measures
- **Secondary Metrics**: Computational efficiency, convergence rate
- **Cross-Validation**: Generalization performance
- **Statistical Significance**: Proper hypothesis testing

## Development

### Testing

The system includes 30+ comprehensive test files:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tests --cov-report=html

# Run specific component tests
pytest tests/test_autonomous_agent.py
pytest tests/test_xpr_agent_engine_comprehensive.py
pytest tests/test_human_layer.py
pytest tests/test_memory_store.py
pytest tests/test_rl_loop.py

# Run integration tests
pytest tests/test_integration_experiments.py

# Run performance and stress tests
pytest tests/test_performance_stress.py

# Run security tests
pytest tests/test_security.py

# GUI tests with Playwright
pytest tests/test_gui_playwright.py
```

### Test Coverage Areas

- **Unit Tests**: Core components (autonomous_agent, xpr_agent_engine, human_layer, memory_store)
- **Integration Tests**: Full RL loop, experiment execution, end-to-end workflows
- **XPR Engine Tests**: Skill chaining, LLM integration, self-healing mechanisms
- **Human Layer Tests**: Review workflow, approval decisions, configuration
- **Memory Tests**: Semantic search, vector embeddings, pattern extraction
- **Performance Tests**: Stress testing, timeout handling, async operations
- **Security Tests**: Input validation, module loading, sandboxed execution
- **GUI Tests**: Playwright-based UI automation and visualization

### Code Quality

```bash
# Linting
flake8 autonomous_agent.py GUI-auto_improve_experiments.py

# Formatting
black *.py

# Type checking
mypy autonomous_agent.py
```

### Documentation

```bash
# Generate API docs
pdoc autonomous_agent.py --html

# Generate coverage report
pytest --cov=tests --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   - Clear Python cache: `python -B -m pytest`
   - Check PYTHONPATH configuration
   - Verify experiment file structure

2. **Git Repository Issues**
   - Initialize git repo: `git init`
   - Check permissions on .git directory
   - Verify remote repository configuration

3. **Performance Issues**
   - Monitor memory usage with `htop`
   - Check GPU availability: `nvidia-smi`
   - Optimize experiment parameters

4. **GUI Issues**
   - Check CustomTkinter installation
   - Verify display server configuration
   - Test with minimal example

### Debug Mode

```bash
# Enable debug logging
python autonomous_agent.py --experiment masking --debug

# GUI debug mode
python GUI-auto_improve_experiments.py --debug
```

## Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure code quality standards
5. Submit pull request

### Code Standards

- **PEP 8** compliance
- **Type hints** for all public APIs
- **Docstrings** for all classes and methods
- **Tests** with >70% coverage
- **Security** review for all changes

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **ALGORITHM Framework** - Autonomous learning algorithm with human-in-the-loop
- **APGI Framework** - Psychological experiment modeling
- **Human Control Layer** - Interactive approval workflow
- **XPR Agent Engine** - LLM-integrated skill execution
- **Memory Store** - Cognitive memory with vector embeddings
- **Hypothesis Approval Board** - Scientific hypothesis management
- **CustomTkinter** - Modern GUI framework
- **GitPython** - Git integration
- **litellm** - Multi-provider LLM API
- **sentence-transformers** - Semantic embeddings
- **PyTorch** - Deep learning backend
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **Playwright** - Browser automation for testing

## Support

For issues, questions, or contributions:

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check README.md and inline docs
- **Community**: Join discussions in GitHub Discussions
