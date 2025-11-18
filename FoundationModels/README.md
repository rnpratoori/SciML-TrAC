# Scientific Machine Learning Tutorial: Aurora and Poseidon Examples

This repository contains comprehensive tutorial examples for two state-of-the-art scientific machine learning models:
- **Aurora**: Foundation model for Earth system forecasting
- **Poseidon**: Efficient foundation model for solving PDEs

## 📚 Tutorial Contents

### Aurora Examples (`aurora_examples.py`)

Aurora is designed for weather and climate prediction. The tutorial includes:

1. **Basic Weather Prediction**: Introduction to Aurora's core functionality for predicting atmospheric variables
2. **Multi-step Forecasting**: Demonstrates rollout techniques for extended forecasts
3. **Physical Consistency Analysis**: Validates predictions against physical laws and constraints
4. **Extreme Weather Events**: Simulates tropical cyclones and other extreme phenomena
5. **Energy and Momentum Analysis**: Examines conservation properties in predictions

### Poseidon Examples (`poseidon_examples.py`)

Poseidon is a versatile PDE solver. The tutorial covers:

1. **Heat Equation**: Classic diffusion problems with energy conservation
2. **Wave Equation**: Acoustic wave propagation and interference
3. **Navier-Stokes Equations**: Incompressible fluid flow and turbulence
4. **Allen-Cahn Equation**: Phase separation and reaction-diffusion systems
5. **Helmholtz Equation**: Wave scattering and frequency-domain problems
6. **Coupled Multi-Physics**: Thermal-fluid-reaction systems with multiple interacting fields

## 🚀 Installation

### Prerequisites
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Installing Aurora
```bash
# Install Aurora
pip install microsoft-aurora

# Or with conda
mamba install microsoft-aurora -c conda-forge
```

### Installing Poseidon
```bash
# Clone the Poseidon repository
git clone https://github.com/camlab-ethz/poseidon.git
cd poseidon
pip install -e .
```

## 📖 Usage

### Running Aurora Examples
```python
# Run all Aurora examples
python aurora_examples.py

# Or import specific examples
from aurora_examples import example_basic_weather_prediction
example_basic_weather_prediction()
```

### Running Poseidon Examples
```python
# Run all Poseidon examples
python poseidon_examples.py

# Or import specific examples
from poseidon_examples import example_navier_stokes
example_navier_stokes()
```

### Using Pretrained Models

#### Aurora
```python
from aurora import AuroraSmallPretrained

# Load pretrained model
model = AuroraSmallPretrained()
model.load_checkpoint()
```

#### Poseidon
```python
from scOT.model import ScOT

# Load pretrained models from HuggingFace Hub
model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")  # Tiny model
# model = ScOT.from_pretrained("camlab-ethz/Poseidon-B")  # Base model
# model = ScOT.from_pretrained("camlab-ethz/Poseidon-L")  # Large model
```

## 🔬 Physics Problems Covered

### Conservation Laws
- Energy conservation in heat diffusion
- Momentum conservation in fluid flow
- Mass conservation (incompressibility)
- Wave energy conservation

### Physical Phenomena
- **Atmospheric Dynamics**: Weather patterns, cyclones, temperature gradients
- **Fluid Dynamics**: Vortices, turbulence, boundary layers
- **Wave Physics**: Propagation, scattering, interference
- **Phase Transitions**: Separation, coarsening, interface dynamics
- **Coupled Systems**: Multi-physics interactions

### Numerical Considerations
- Stability analysis
- Physical consistency checks
- Conservation verification
- Boundary condition handling

## 📊 Example Output

Each example provides detailed analysis including:
- Initial and final state statistics
- Conservation law verification
- Physical property calculations
- Error metrics and residuals

## 🎯 Learning Objectives

After completing this tutorial, you will understand:

1. **Model Architecture**: How transformer-based models handle spatiotemporal data
2. **Physical Constraints**: Incorporating physics into ML predictions
3. **Multi-scale Modeling**: Handling phenomena at different spatial and temporal scales
4. **Conservation Properties**: Ensuring physical laws are respected
5. **Practical Applications**: Real-world use cases in weather, fluids, and materials

## 📝 Key Concepts

### Aurora
- **Foundation Models**: Pre-trained on diverse atmospheric data
- **Rollout Strategies**: Multi-step predictions with error accumulation
- **Physical Variables**: Temperature, pressure, wind, humidity
- **Temporal Context**: Using historical data for predictions

### Poseidon
- **PDE Types**: Elliptic, parabolic, hyperbolic equations
- **Boundary Conditions**: Periodic, Dirichlet, Neumann
- **Spectral Methods**: Fourier-based approaches for certain PDEs
- **Operator Learning**: Mapping between function spaces

## 🔧 Customization

### Modifying Examples
Each example can be customized by adjusting:
- Grid resolution
- Time steps
- Physical parameters (viscosity, diffusivity, etc.)
- Initial conditions
- Boundary conditions

### Adding New Physics
To add new physics problems:
1. Define the PDE and physical parameters
2. Create appropriate initial conditions
3. Implement physical consistency checks
4. Add visualization routines

## 📚 References

### Aurora
- Paper: [A Foundation Model for the Earth System](https://arxiv.org/abs/2405.13063)
- GitHub: [microsoft/aurora](https://github.com/microsoft/aurora)
- Documentation: [Aurora Docs](https://microsoft.github.io/aurora)

### Poseidon
- Paper: [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101)
- GitHub: [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon)
- Models: [HuggingFace Hub](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new physics examples
- Improve existing demonstrations
- Add visualization capabilities
- Enhance documentation

## 📄 License

Please refer to the original repositories for licensing information:
- Aurora: Microsoft's license terms
- Poseidon: ETH Zurich's license terms

## 💡 Tips for Best Results

1. **Data Preprocessing**: Normalize your data appropriately
2. **Model Selection**: Choose model size based on problem complexity
3. **Batch Processing**: Use batching for efficiency
4. **GPU Acceleration**: Ensure CUDA is properly configured
5. **Memory Management**: Monitor GPU memory for large problems

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller grid resolution
   - Use gradient checkpointing

2. **Model Download Issues**
   - Check internet connection
   - Verify HuggingFace Hub access
   - Use local cache if available

3. **Convergence Problems**
   - Check initial conditions
   - Verify physical parameters
   - Adjust time step size

## 📈 Performance Considerations

- **Aurora**: Optimized for 6-hour weather prediction steps
- **Poseidon**: Scales well with transformer architecture
- Both models benefit from GPU acceleration
- Batch processing improves throughput

## 🎓 Educational Use

This tutorial is designed for:
- Graduate students in computational physics
- Researchers in scientific computing
- Engineers working on physics simulations
- Data scientists interested in physical modeling

## 🔮 Future Directions

Potential extensions include:
- Multi-GPU distributed training
- Real-time inference optimization
- Uncertainty quantification
- Hybrid physics-ML approaches
- Domain adaptation techniques

---

**Note**: These examples are for educational purposes. For production use, please refer to the official documentation and consider additional validation and testing.
