# APGI Research Application - Program Overview

## Introduction

The APGI Research Application is a sophisticated autonomous research system designed to optimize psychological experiments through AI-driven parameter tuning and continuous learning.

## Architecture

### Core Components

1. **AutonomousAgent** - Main controller orchestrating optimization loops
2. **GitPerformanceTracker** - Version control-based performance tracking
3. **ParameterOptimizer** - AI-driven parameter modification algorithms
4. **ExperimentRunnerGUI** - Modern GUI for experiment management
5. **APGI Integration** - Psychological experiment modeling framework

### Key Features

- **Infinite Autonomous Optimization**: Continuous parameter optimization using AI agents
- **Git-Based Learning**: Track performance improvements and rollback failures
- **Multi-Experiment Support**: Optimize across 30+ psychological experiments
- **Real-Time Monitoring**: GUI with live experiment execution and results
- **Cross-Platform Compatibility**: Works on macOS, Linux, and Windows
- **Security Hardened**: Input validation, sandboxed execution, and secure module loading

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

# Initialize agent
agent = AutonomousAgent()

# Run optimization
results = agent.optimize_experiment("attentional_blink", iterations=100)

# Run single experiment
result = agent.run_experiment("masking", modifications={"BASE_DETECTION_RATE": 0.5})
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

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tests --cov-report=html

# Run specific test
pytest tests/test_autonomous_agent.py::TestAutonomousAgent::test_parameter_extraction
```

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

- **APGI Framework** - Psychological experiment modeling
- **CustomTkinter** - Modern GUI framework
- **GitPython** - Git integration
- **PyTorch** - Deep learning backend
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization

## Support

For issues, questions, or contributions:

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check README.md and inline docs
- **Community**: Join discussions in GitHub Discussions

---

Last Updated: March 22, 2026
