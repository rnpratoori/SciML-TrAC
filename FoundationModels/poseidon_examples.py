"""
Poseidon Examples for Scientific Machine Learning Tutorial
===========================================================

Poseidon is an efficient foundation model for solving Partial Differential
Equations (PDEs) across various physics domains.

This tutorial demonstrates several physics problems using Poseidon.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scOT.model import ScOT, ScOTConfig
from typing import Dict, Tuple, Optional
import torch.nn.functional as F


# ============================================================================
# Example 1: Heat Equation (Diffusion)
# ============================================================================
def example_heat_equation():
    """
    Solves the 2D heat equation using Poseidon.
    
    ∂u/∂t = α∇²u
    
    where u is temperature and α is thermal diffusivity.
    """
    print("=" * 60)
    print("Example 1: Heat Equation (Diffusion Process)")
    print("=" * 60)
    
    # Model configuration for heat equation
    config = ScOTConfig(
        input_channels=1,  # Initial temperature field
        output_channels=1,  # Final temperature field
        image_size=64,
        patch_size=4,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
    )
    
    # Initialize model (would normally load pretrained)
    # For demonstration, we're showing the structure
    model = ScOT(config)
    model.eval()
    
    # Create initial condition: Gaussian heat source
    batch_size = 4
    grid_size = 64
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Multiple heat sources
    initial_temp = torch.zeros(batch_size, 1, grid_size, grid_size)
    
    for b in range(batch_size):
        # Random heat source positions
        n_sources = torch.randint(1, 4, (1,)).item()
        for _ in range(n_sources):
            x_center = torch.rand(1).item() * 1.6 - 0.8
            y_center = torch.rand(1).item() * 1.6 - 0.8
            amplitude = torch.rand(1).item() * 0.5 + 0.5
            width = torch.rand(1).item() * 0.1 + 0.05
            
            heat_source = amplitude * torch.exp(
                -((X - x_center)**2 + (Y - y_center)**2) / (2 * width**2)
            )
            initial_temp[b, 0] += heat_source
    
    # Time embedding (normalized time)
    time_steps = torch.tensor([0.0, 0.25, 0.5, 1.0])  # Different time points
    
    print(f"Initial condition shape: {initial_temp.shape}")
    print(f"Initial temperature range: [{initial_temp.min():.3f}, {initial_temp.max():.3f}]")
    
    # Simulate forward pass (in practice, would use pretrained model)
    with torch.no_grad():
        # The model would predict the temperature field at different times
        # Here we simulate the diffusion behavior
        predicted_temp = simulate_heat_diffusion(initial_temp, time_steps[-1])
    
    print(f"Final temperature range: [{predicted_temp.min():.3f}, {predicted_temp.max():.3f}]")
    
    # Verify physical properties
    print("\nPhysical Consistency Checks:")
    print(f"1. Energy conservation: Initial sum = {initial_temp.sum():.3f}, "
          f"Final sum = {predicted_temp.sum():.3f}")
    print(f"2. Maximum principle: Max decreased from {initial_temp.max():.3f} "
          f"to {predicted_temp.max():.3f}")
    print(f"3. Smoothness: Initial gradient norm = {compute_gradient_norm(initial_temp):.3f}, "
          f"Final = {compute_gradient_norm(predicted_temp):.3f}")
    print()


def simulate_heat_diffusion(u: torch.Tensor, t: float, alpha: float = 0.1) -> torch.Tensor:
    """Simple heat diffusion simulation for demonstration."""
    # Apply Gaussian filter to simulate diffusion
    sigma = torch.sqrt(2 * alpha * t)
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    x = torch.arange(kernel_size).float() - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution
    u_padded = F.pad(u, (kernel_size//2,) * 4, mode='reflect')
    u_diffused = F.conv2d(u_padded, kernel, padding=0)
    
    return u_diffused


def compute_gradient_norm(u: torch.Tensor) -> float:
    """Compute the norm of spatial gradients."""
    dx = u[:, :, 1:, :] - u[:, :, :-1, :]
    dy = u[:, :, :, 1:] - u[:, :, :, :-1]
    return (dx**2).mean().sqrt().item() + (dy**2).mean().sqrt().item()


# ============================================================================
# Example 2: Wave Equation
# ============================================================================
def example_wave_equation():
    """
    Solves the 2D wave equation using Poseidon.
    
    ∂²u/∂t² = c²∇²u
    
    where u is displacement and c is wave speed.
    """
    print("=" * 60)
    print("Example 2: Wave Equation (Acoustic Waves)")
    print("=" * 60)
    
    config = ScOTConfig(
        input_channels=2,  # Initial displacement and velocity
        output_channels=2,  # Final displacement and velocity
        image_size=64,
        patch_size=4,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
    )
    
    model = ScOT(config)
    model.eval()
    
    # Create initial conditions
    batch_size = 2
    grid_size = 64
    
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Initial displacement (Gaussian pulse)
    u0 = torch.zeros(batch_size, 1, grid_size, grid_size)
    v0 = torch.zeros(batch_size, 1, grid_size, grid_size)  # Initial velocity
    
    for b in range(batch_size):
        # Create different wave patterns
        if b == 0:
            # Single pulse
            u0[b, 0] = 0.5 * torch.exp(-20 * (X**2 + Y**2))
        else:
            # Two interfering pulses
            u0[b, 0] = (0.3 * torch.exp(-20 * ((X - 0.3)**2 + Y**2)) +
                       0.3 * torch.exp(-20 * ((X + 0.3)**2 + Y**2)))
    
    initial_state = torch.cat([u0, v0], dim=1)
    
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Initial displacement range: [{u0.min():.3f}, {u0.max():.3f}]")
    
    # Simulate wave propagation
    with torch.no_grad():
        # Model would predict the wave field at later times
        predicted_state = simulate_wave_propagation(initial_state, t=1.0)
    
    u_final = predicted_state[:, 0:1]
    v_final = predicted_state[:, 1:2]
    
    print(f"Final displacement range: [{u_final.min():.3f}, {u_final.max():.3f}]")
    
    # Calculate wave properties
    print("\nWave Properties:")
    
    # Energy calculation
    kinetic_energy_init = 0.5 * (v0**2).sum()
    potential_energy_init = 0.5 * compute_wave_potential_energy(u0)
    total_energy_init = kinetic_energy_init + potential_energy_init
    
    kinetic_energy_final = 0.5 * (v_final**2).sum()
    potential_energy_final = 0.5 * compute_wave_potential_energy(u_final)
    total_energy_final = kinetic_energy_final + potential_energy_final
    
    print(f"1. Initial total energy: {total_energy_init:.3f}")
    print(f"2. Final total energy: {total_energy_final:.3f}")
    print(f"3. Energy conservation error: {abs(total_energy_final - total_energy_init):.6f}")
    print()


def simulate_wave_propagation(state: torch.Tensor, t: float, c: float = 1.0) -> torch.Tensor:
    """Simple wave propagation for demonstration."""
    u = state[:, 0:1]
    v = state[:, 1:2]
    
    # Use simple finite difference approximation
    dt = 0.01
    steps = int(t / dt)
    
    for _ in range(steps):
        # Compute Laplacian
        laplacian = compute_laplacian(u)
        
        # Update velocity
        v = v + c**2 * laplacian * dt
        
        # Update displacement
        u = u + v * dt
    
    return torch.cat([u, v], dim=1)


def compute_laplacian(u: torch.Tensor) -> torch.Tensor:
    """Compute 2D Laplacian using finite differences."""
    # Simple 5-point stencil
    kernel = torch.tensor([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=u.dtype).view(1, 1, 3, 3)
    
    u_padded = F.pad(u, (1, 1, 1, 1), mode='reflect')
    return F.conv2d(u_padded, kernel, padding=0)


def compute_wave_potential_energy(u: torch.Tensor) -> torch.Tensor:
    """Compute potential energy in the wave field."""
    dx = u[:, :, 1:, :] - u[:, :, :-1, :]
    dy = u[:, :, :, 1:] - u[:, :, :, :-1]
    return (dx**2).sum() + (dy**2).sum()


# ============================================================================
# Example 3: Navier-Stokes Equations (Fluid Flow)
# ============================================================================
def example_navier_stokes():
    """
    Solves 2D incompressible Navier-Stokes equations using Poseidon.
    
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u
    ∇·u = 0
    
    where u is velocity, p is pressure, ν is kinematic viscosity.
    """
    print("=" * 60)
    print("Example 3: Navier-Stokes Equations (Fluid Flow)")
    print("=" * 60)
    
    config = ScOTConfig(
        input_channels=3,  # u, v velocity components and vorticity
        output_channels=3,
        image_size=128,
        patch_size=8,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    )
    
    model = ScOT(config)
    model.eval()
    
    # Create initial vorticity field for different flow patterns
    batch_size = 3
    grid_size = 128
    
    x = torch.linspace(0, 2*np.pi, grid_size)
    y = torch.linspace(0, 2*np.pi, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Different flow patterns
    initial_fields = torch.zeros(batch_size, 3, grid_size, grid_size)
    
    # Pattern 1: Taylor-Green vortex
    initial_fields[0, 0] = torch.sin(X) * torch.cos(Y)  # u velocity
    initial_fields[0, 1] = -torch.cos(X) * torch.sin(Y)  # v velocity
    initial_fields[0, 2] = 2 * torch.sin(X) * torch.sin(Y)  # vorticity
    
    # Pattern 2: Random turbulence
    torch.manual_seed(42)
    for k in range(1, 10):
        amplitude = 1.0 / k
        phase_x = torch.rand(1) * 2 * np.pi
        phase_y = torch.rand(1) * 2 * np.pi
        initial_fields[1, 2] += amplitude * torch.sin(k * X + phase_x) * torch.sin(k * Y + phase_y)
    
    # Pattern 3: Vortex pair
    vortex1_x, vortex1_y = np.pi - 0.5, np.pi
    vortex2_x, vortex2_y = np.pi + 0.5, np.pi
    r1 = torch.sqrt((X - vortex1_x)**2 + (Y - vortex1_y)**2)
    r2 = torch.sqrt((X - vortex2_x)**2 + (Y - vortex2_y)**2)
    initial_fields[2, 2] = (torch.exp(-2 * r1**2) - torch.exp(-2 * r2**2))
    
    print(f"Initial flow field shape: {initial_fields.shape}")
    
    # Physical parameters
    reynolds_number = 1000
    viscosity = 1.0 / reynolds_number
    
    print(f"Reynolds number: {reynolds_number}")
    print(f"Kinematic viscosity: {viscosity:.4f}")
    
    # Simulate fluid evolution
    with torch.no_grad():
        # Model would predict evolved flow field
        predicted_fields = simulate_navier_stokes(initial_fields, t=1.0, nu=viscosity)
    
    # Analyze flow properties
    print("\nFlow Analysis:")
    
    # Compute kinetic energy
    u_init = initial_fields[:, 0]
    v_init = initial_fields[:, 1]
    kinetic_energy_init = 0.5 * (u_init**2 + v_init**2).mean()
    
    u_final = predicted_fields[:, 0]
    v_final = predicted_fields[:, 1]
    kinetic_energy_final = 0.5 * (u_final**2 + v_final**2).mean()
    
    print(f"1. Initial kinetic energy: {kinetic_energy_init:.4f}")
    print(f"2. Final kinetic energy: {kinetic_energy_final:.4f}")
    print(f"3. Energy dissipation: {(kinetic_energy_init - kinetic_energy_final):.4f}")
    
    # Compute enstrophy (vorticity squared)
    vorticity_init = initial_fields[:, 2]
    vorticity_final = predicted_fields[:, 2]
    enstrophy_init = (vorticity_init**2).mean()
    enstrophy_final = (vorticity_final**2).mean()
    
    print(f"4. Initial enstrophy: {enstrophy_init:.4f}")
    print(f"5. Final enstrophy: {enstrophy_final:.4f}")
    
    # Check incompressibility (divergence should be zero)
    divergence = compute_divergence(u_final, v_final)
    print(f"6. Divergence (should be ~0): {divergence.abs().mean():.6f}")
    print()


def simulate_navier_stokes(fields: torch.Tensor, t: float, nu: float) -> torch.Tensor:
    """Simplified Navier-Stokes simulation."""
    # For demonstration, apply viscous diffusion to vorticity
    vorticity = fields[:, 2:3]
    
    # Diffuse vorticity
    vorticity_diffused = simulate_heat_diffusion(vorticity, t, alpha=nu)
    
    # Recover velocity from vorticity (simplified)
    u = fields[:, 0:1] * torch.exp(-nu * t)
    v = fields[:, 1:2] * torch.exp(-nu * t)
    
    return torch.cat([u, v, vorticity_diffused], dim=1)


def compute_divergence(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute velocity divergence."""
    du_dx = (u[:, 1:, :] - u[:, :-1, :]).mean(dim=0)
    dv_dy = (v[:, :, 1:] - v[:, :, :-1]).mean(dim=0)
    return du_dx[:-1, :] + dv_dy[:, :-1]


# ============================================================================
# Example 4: Reaction-Diffusion (Allen-Cahn Equation)
# ============================================================================
def example_reaction_diffusion():
    """
    Solves the Allen-Cahn equation, a reaction-diffusion PDE.
    
    ∂u/∂t = ε²∇²u - (u³ - u)
    
    Models phase separation in materials.
    """
    print("=" * 60)
    print("Example 4: Allen-Cahn Equation (Phase Separation)")
    print("=" * 60)
    
    config = ScOTConfig(
        input_channels=1,  # Phase field
        output_channels=1,
        image_size=128,
        patch_size=8,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
    )
    
    model = ScOT(config)
    model.eval()
    
    batch_size = 4
    grid_size = 128
    
    # Create initial conditions with random perturbations
    initial_phase = torch.zeros(batch_size, 1, grid_size, grid_size)
    
    for b in range(batch_size):
        # Different initial patterns
        if b == 0:
            # Random noise around zero
            initial_phase[b] = 0.1 * torch.randn(1, grid_size, grid_size)
        elif b == 1:
            # Circular inclusion
            x = torch.linspace(-1, 1, grid_size)
            y = torch.linspace(-1, 1, grid_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            r = torch.sqrt(X**2 + Y**2)
            initial_phase[b, 0] = torch.tanh(10 * (0.5 - r))
        elif b == 2:
            # Striped pattern
            x = torch.linspace(0, 4*np.pi, grid_size)
            initial_phase[b, 0] = torch.sin(x).unsqueeze(1).expand(grid_size, grid_size)
        else:
            # Multiple bubbles
            initial_phase[b] = -1.0
            for _ in range(5):
                cx = torch.rand(1) * 2 - 1
                cy = torch.rand(1) * 2 - 1
                x = torch.linspace(-1, 1, grid_size)
                y = torch.linspace(-1, 1, grid_size)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                r = torch.sqrt((X - cx)**2 + (Y - cy)**2)
                initial_phase[b, 0] = torch.maximum(initial_phase[b, 0], 
                                                   torch.tanh(20 * (0.2 - r)))
    
    print(f"Initial phase field shape: {initial_phase.shape}")
    print(f"Initial phase range: [{initial_phase.min():.3f}, {initial_phase.max():.3f}]")
    
    # Interface width parameter
    epsilon = 0.05
    
    # Simulate phase evolution
    with torch.no_grad():
        predicted_phase = simulate_allen_cahn(initial_phase, t=10.0, epsilon=epsilon)
    
    print(f"Final phase range: [{predicted_phase.min():.3f}, {predicted_phase.max():.3f}]")
    
    # Analyze phase separation
    print("\nPhase Separation Analysis:")
    
    # Compute interface length (gradient magnitude)
    grad_init = compute_gradient_magnitude(initial_phase)
    grad_final = compute_gradient_magnitude(predicted_phase)
    interface_length_init = grad_init.sum()
    interface_length_final = grad_final.sum()
    
    print(f"1. Initial interface length: {interface_length_init:.1f}")
    print(f"2. Final interface length: {interface_length_final:.1f}")
    print(f"3. Interface reduction: {(interface_length_init - interface_length_final):.1f}")
    
    # Compute volume fractions
    volume_positive_init = (initial_phase > 0).float().mean()
    volume_positive_final = (predicted_phase > 0).float().mean()
    
    print(f"4. Initial positive phase fraction: {volume_positive_init:.3f}")
    print(f"5. Final positive phase fraction: {volume_positive_final:.3f}")
    
    # Energy calculation
    energy_init = compute_allen_cahn_energy(initial_phase, epsilon)
    energy_final = compute_allen_cahn_energy(predicted_phase, epsilon)
    
    print(f"6. Initial energy: {energy_init:.3f}")
    print(f"7. Final energy: {energy_final:.3f}")
    print(f"8. Energy reduction: {(energy_init - energy_final):.3f}")
    print()


def simulate_allen_cahn(u: torch.Tensor, t: float, epsilon: float) -> torch.Tensor:
    """Simulate Allen-Cahn equation evolution."""
    dt = 0.001
    steps = int(t / dt)
    
    for _ in range(min(steps, 100)):  # Limit steps for demonstration
        laplacian = compute_laplacian(u)
        reaction = u - u**3
        u = u + dt * (epsilon**2 * laplacian + reaction)
    
    return torch.tanh(u / epsilon)  # Sharp interface


def compute_gradient_magnitude(u: torch.Tensor) -> torch.Tensor:
    """Compute magnitude of gradient."""
    dx = u[:, :, 1:, :] - u[:, :, :-1, :]
    dy = u[:, :, :, 1:] - u[:, :, :, :-1]
    return torch.sqrt(dx[:, :, :, :-1]**2 + dy[:, :, :-1, :]**2)


def compute_allen_cahn_energy(u: torch.Tensor, epsilon: float) -> float:
    """Compute Allen-Cahn free energy."""
    grad_mag = compute_gradient_magnitude(u)
    gradient_energy = 0.5 * epsilon**2 * (grad_mag**2).sum()
    potential_energy = 0.25 * ((u**2 - 1)**2).sum()
    return (gradient_energy + potential_energy).item()


# ============================================================================
# Example 5: Helmholtz Equation (Wave Scattering)
# ============================================================================
def example_helmholtz():
    """
    Solves the Helmholtz equation for wave scattering problems.
    
    ∇²u + k²u = f
    
    where u is the wave field, k is the wave number, f is the source.
    """
    print("=" * 60)
    print("Example 5: Helmholtz Equation (Wave Scattering)")
    print("=" * 60)
    
    config = ScOTConfig(
        input_channels=2,  # Real and imaginary parts of source
        output_channels=2,  # Real and imaginary parts of solution
        image_size=128,
        patch_size=8,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
    )
    
    model = ScOT(config)
    model.eval()
    
    batch_size = 3
    grid_size = 128
    
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Different scattering scenarios
    sources = torch.zeros(batch_size, 2, grid_size, grid_size)
    
    # Wave numbers (frequency)
    k_values = [2.0, 4.0, 6.0]
    
    for b in range(batch_size):
        k = k_values[b]
        
        if b == 0:
            # Point source
            cx, cy = 0, 0
            r = torch.sqrt((X - cx)**2 + (Y - cy)**2)
            sources[b, 0] = torch.exp(-10 * r**2)  # Real part
            
        elif b == 1:
            # Plane wave with obstacle
            sources[b, 0] = torch.cos(k * X)  # Real part
            sources[b, 1] = torch.sin(k * X)  # Imaginary part
            
            # Add circular obstacle (zero source inside)
            r = torch.sqrt(X**2 + Y**2)
            mask = r < 1.0
            sources[b, 0][mask] = 0
            sources[b, 1][mask] = 0
            
        else:
            # Multiple point sources (interference pattern)
            positions = [(2, 0), (-2, 0), (0, 2)]
            for px, py in positions:
                r = torch.sqrt((X - px)**2 + (Y - py)**2)
                phase = k * r
                sources[b, 0] += torch.cos(phase) * torch.exp(-0.5 * r**2)
                sources[b, 1] += torch.sin(phase) * torch.exp(-0.5 * r**2)
    
    print(f"Source field shape: {sources.shape}")
    print(f"Wave numbers: {k_values}")
    
    # Solve Helmholtz equation
    with torch.no_grad():
        # Model would solve for the wave field
        wave_field = solve_helmholtz(sources, k_values)
    
    # Analyze wave properties
    print("\nWave Field Analysis:")
    
    for b, k in enumerate(k_values):
        real_part = wave_field[b, 0]
        imag_part = wave_field[b, 1]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)
        
        print(f"\nConfiguration {b+1} (k={k}):")
        print(f"  Magnitude range: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
        print(f"  Phase range: [{phase.min():.3f}, {phase.max():.3f}]")
        
        # Check solution quality (residual)
        residual = compute_helmholtz_residual(wave_field[b:b+1], sources[b:b+1], k)
        print(f"  Residual norm: {residual:.6f}")
    
    # Compute scattering cross-section for obstacle case
    if batch_size > 1:
        print("\nScattering Analysis (Configuration 2):")
        scattered_field = wave_field[1] - sources[1]
        scattered_magnitude = torch.sqrt(scattered_field[0]**2 + scattered_field[1]**2)
        
        # Far-field pattern (sample along circle)
        angles = torch.linspace(0, 2*np.pi, 100)
        radius = 4.0
        far_field = []
        for angle in angles:
            x_pos = int(grid_size/2 + radius * torch.cos(angle) * grid_size/10)
            y_pos = int(grid_size/2 + radius * torch.sin(angle) * grid_size/10)
            if 0 <= x_pos < grid_size and 0 <= y_pos < grid_size:
                far_field.append(scattered_magnitude[x_pos, y_pos].item())
        
        if far_field:
            print(f"  Far-field amplitude range: [{min(far_field):.3f}, {max(far_field):.3f}]")
    
    print()


def solve_helmholtz(sources: torch.Tensor, k_values: list) -> torch.Tensor:
    """Approximate Helmholtz solution for demonstration."""
    batch_size, _, h, w = sources.shape
    solution = torch.zeros_like(sources)
    
    for b, k in enumerate(k_values):
        # Simple Green's function approach (2D)
        source_real = sources[b, 0]
        source_imag = sources[b, 1]
        
        # Convolve with Green's function (simplified)
        x = torch.linspace(-5, 5, h)
        y = torch.linspace(-5, 5, w)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(X**2 + Y**2)
        r[r < 0.1] = 0.1  # Avoid singularity
        
        # Approximate Green's function
        green_real = torch.cos(k * r) / (4 * np.pi * r)
        green_imag = torch.sin(k * r) / (4 * np.pi * r)
        
        # Apply (simplified convolution)
        solution[b, 0] = source_real * green_real - source_imag * green_imag
        solution[b, 1] = source_real * green_imag + source_imag * green_real
    
    return solution


def compute_helmholtz_residual(u: torch.Tensor, f: torch.Tensor, k: float) -> float:
    """Compute residual of Helmholtz equation."""
    laplacian_real = compute_laplacian(u[:, 0:1])
    laplacian_imag = compute_laplacian(u[:, 1:2])
    
    residual_real = laplacian_real + k**2 * u[:, 0:1] - f[:, 0:1]
    residual_imag = laplacian_imag + k**2 * u[:, 1:2] - f[:, 1:2]
    
    return torch.sqrt((residual_real**2 + residual_imag**2).mean()).item()


# ============================================================================
# Example 6: Coupled Multi-Physics Problem
# ============================================================================
def example_coupled_physics():
    """
    Demonstrates Poseidon on a coupled multi-physics problem:
    Thermal-fluid interaction with reaction.
    """
    print("=" * 60)
    print("Example 6: Coupled Multi-Physics (Thermal-Fluid-Reaction)")
    print("=" * 60)
    
    config = ScOTConfig(
        input_channels=5,  # u, v, temperature, concentration, pressure
        output_channels=5,
        image_size=128,
        patch_size=8,
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=12,
    )
    
    model = ScOT(config)
    model.eval()
    
    batch_size = 2
    grid_size = 128
    
    # Initialize coupled fields
    fields = torch.zeros(batch_size, 5, grid_size, grid_size)
    
    x = torch.linspace(0, 2*np.pi, grid_size)
    y = torch.linspace(0, 2*np.pi, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    for b in range(batch_size):
        # Velocity field (convection)
        fields[b, 0] = 0.5 * torch.sin(X) * torch.cos(Y)  # u
        fields[b, 1] = -0.5 * torch.cos(X) * torch.sin(Y)  # v
        
        # Temperature field (affects reaction rate)
        if b == 0:
            fields[b, 2] = 300 + 20 * torch.exp(-((X - np.pi)**2 + (Y - np.pi)**2) / 2)
        else:
            fields[b, 2] = 300 + 10 * (torch.sin(X) + torch.sin(Y))
        
        # Concentration field (chemical species)
        fields[b, 3] = 0.5 + 0.3 * torch.cos(2*X) * torch.cos(2*Y)
        
        # Pressure field
        fields[b, 4] = 101325 + 100 * torch.sin(X + Y)
    
    print(f"Coupled system shape: {fields.shape}")
    print("Field components: [u_velocity, v_velocity, temperature, concentration, pressure]")
    
    # Physical parameters
    print("\nPhysical Parameters:")
    print("  Thermal diffusivity: 0.01")
    print("  Species diffusivity: 0.005")
    print("  Reaction rate constant: 0.1")
    print("  Activation energy: 10 kJ/mol")
    
    # Simulate coupled evolution
    with torch.no_grad():
        evolved_fields = simulate_coupled_system(fields, t=1.0)
    
    # Analyze coupling effects
    print("\nCoupling Analysis:")
    
    for b in range(batch_size):
        print(f"\nConfiguration {b+1}:")
        
        # Temperature statistics
        temp_init = fields[b, 2]
        temp_final = evolved_fields[b, 2]
        print(f"  Temperature: {temp_init.mean():.1f} → {temp_final.mean():.1f} K")
        
        # Concentration statistics
        conc_init = fields[b, 3]
        conc_final = evolved_fields[b, 3]
        print(f"  Concentration: {conc_init.mean():.3f} → {conc_final.mean():.3f}")
        
        # Reaction rate (Arrhenius)
        k_reaction = 0.1 * torch.exp(-10000 / (8.314 * temp_final))
        reaction_rate = k_reaction * conc_final
        print(f"  Mean reaction rate: {reaction_rate.mean():.4f}")
        
        # Transport effects
        peclet_number = fields[b, 0].abs().mean() * grid_size / 0.01
        print(f"  Péclet number: {peclet_number:.1f}")
        
        # Energy balance
        thermal_energy = temp_final.sum()
        chemical_energy = -conc_final.sum() * 50000  # Exothermic reaction
        print(f"  Thermal energy: {thermal_energy:.0f}")
        print(f"  Chemical energy: {chemical_energy:.0f}")
    
    print()


def simulate_coupled_system(fields: torch.Tensor, t: float) -> torch.Tensor:
    """Simulate coupled thermal-fluid-reaction system."""
    dt = 0.01
    steps = int(t / dt)
    
    u = fields[:, 0:1]
    v = fields[:, 1:2]
    temp = fields[:, 2:3]
    conc = fields[:, 3:4]
    pressure = fields[:, 4:5]
    
    for _ in range(min(steps, 50)):
        # Advection
        temp_adv = compute_advection(temp, u, v)
        conc_adv = compute_advection(conc, u, v)
        
        # Diffusion
        temp_diff = 0.01 * compute_laplacian(temp)
        conc_diff = 0.005 * compute_laplacian(conc)
        
        # Reaction (temperature-dependent)
        k_reaction = 0.1 * torch.exp(-10000 / (8.314 * temp))
        reaction = -k_reaction * conc
        heat_release = -reaction * 50000 / 1000  # Heat of reaction
        
        # Update fields
        temp = temp + dt * (-temp_adv + temp_diff + heat_release)
        conc = conc + dt * (-conc_adv + conc_diff + reaction)
    
    return torch.cat([u, v, temp, conc, pressure], dim=1)


def compute_advection(field: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute advection term u·∇field."""
    # Simplified upwind scheme
    dx = (field[:, :, 2:, :] - field[:, :, :-2, :]) / 2
    dy = (field[:, :, :, 2:] - field[:, :, :, :-2]) / 2
    
    # Pad to match dimensions
    dx = F.pad(dx, (0, 0, 1, 1), mode='replicate')
    dy = F.pad(dy, (1, 1, 0, 0), mode='replicate')
    
    return u * dx + v * dy


# ============================================================================
# Utility Functions for Visualization
# ============================================================================
def plot_field_comparison(initial: torch.Tensor, final: torch.Tensor, 
                          title: str = "Field Evolution"):
    """Helper function to visualize field evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(initial.squeeze().numpy(), cmap='RdBu_r')
    axes[0].set_title("Initial")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(final.squeeze().numpy(), cmap='RdBu_r')
    axes[1].set_title("Final")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Poseidon Tutorial - Scientific Machine Learning for PDEs")
    print("=" * 60 + "\n")
    
    # Note: To use pretrained models, replace model initialization with:
    # model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")  # or -B, -L
    
    # Run all examples
    example_heat_equation()
    example_wave_equation()
    example_navier_stokes()
    example_reaction_diffusion()
    example_helmholtz()
    example_coupled_physics()
    
    print("=" * 60)
    print("Tutorial completed successfully!")
    print("Note: For production use, load pretrained models from HuggingFace Hub")
    print("=" * 60)
