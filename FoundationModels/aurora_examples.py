"""
Aurora Examples for Scientific Machine Learning Tutorial
=========================================================

Aurora is a foundation model for Earth system forecasting that can predict
atmospheric variables like temperature, pressure, wind, and more.

This tutorial demonstrates several physics problems using Aurora.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from aurora import AuroraSmallPretrained, Batch, Metadata
import xarray as xr


# ============================================================================
# Example 1: Basic Weather Prediction
# ============================================================================
def example_basic_weather_prediction():
    """
    Demonstrates basic weather forecasting using Aurora.
    This example shows how to make predictions for temperature, wind, and pressure.
    """
    print("=" * 60)
    print("Example 1: Basic Weather Prediction with Aurora")
    print("=" * 60)
    
    # Initialize the model
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()
    
    # Create sample initial conditions
    # Dimensions: [batch, time, lat, lon]
    batch_size = 1
    n_times = 2  # Initial time + 1 previous for temporal context
    n_lat = 17
    n_lon = 32
    n_levels = 4
    
    # Surface variables
    surf_vars = {
        "2t": torch.randn(batch_size, n_times, n_lat, n_lon) * 10 + 288,  # 2m temperature
        "10u": torch.randn(batch_size, n_times, n_lat, n_lon) * 5,        # 10m u-wind
        "10v": torch.randn(batch_size, n_times, n_lat, n_lon) * 5,        # 10m v-wind
        "msl": torch.randn(batch_size, n_times, n_lat, n_lon) * 100 + 101325,  # Mean sea level pressure
    }
    
    # Static variables (time-independent)
    static_vars = {
        "lsm": torch.rand(n_lat, n_lon),  # Land-sea mask
        "z": torch.randn(n_lat, n_lon) * 1000,  # Geopotential
        "slt": torch.rand(n_lat, n_lon) * 7,  # Soil type
    }
    
    # Atmospheric variables at different pressure levels
    atmos_vars = {
        "z": torch.randn(batch_size, n_times, n_levels, n_lat, n_lon) * 1000,  # Geopotential height
        "u": torch.randn(batch_size, n_times, n_levels, n_lat, n_lon) * 20,    # u-wind
        "v": torch.randn(batch_size, n_times, n_levels, n_lat, n_lon) * 20,    # v-wind
        "t": torch.randn(batch_size, n_times, n_levels, n_lat, n_lon) * 20 + 250,  # Temperature
        "q": torch.rand(batch_size, n_times, n_levels, n_lat, n_lon) * 0.02,  # Specific humidity
    }
    
    # Create batch with metadata
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=torch.linspace(90, -90, n_lat),
            lon=torch.linspace(0, 360, n_lon + 1)[:-1],
            time=(datetime(2024, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),  # Pressure levels in hPa
        ),
    )
    
    # Make prediction
    with torch.no_grad():
        prediction = model.forward(batch)
    
    print(f"Input shape (2m temperature): {surf_vars['2t'].shape}")
    print(f"Output shape (2m temperature): {prediction.surf_vars['2t'].shape}")
    print(f"Predicted 2m temperature stats:")
    print(f"  Mean: {prediction.surf_vars['2t'].mean().item():.2f} K")
    print(f"  Std:  {prediction.surf_vars['2t'].std().item():.2f} K")
    print()


# ============================================================================
# Example 2: Multi-step Forecasting (Rollout)
# ============================================================================
def example_multistep_forecast():
    """
    Demonstrates how to perform multi-step forecasting by rolling out
    the model predictions iteratively.
    """
    print("=" * 60)
    print("Example 2: Multi-step Weather Forecasting")
    print("=" * 60)
    
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()
    
    # Initialize with more realistic patterns
    n_lat, n_lon = 17, 32
    n_levels = 4
    n_forecast_steps = 10
    
    # Create initial conditions with a temperature wave pattern
    lat = torch.linspace(90, -90, n_lat)
    lon = torch.linspace(0, 360, n_lon + 1)[:-1]
    
    # Create a sinusoidal temperature pattern
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
    temp_pattern = 288 + 10 * torch.sin(2 * np.pi * lon_grid / 360) * torch.cos(lat_grid * np.pi / 180)
    
    # Store predictions
    predictions = []
    
    # Initial conditions
    current_surf_vars = {
        "2t": temp_pattern.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "10u": torch.randn(1, 2, n_lat, n_lon) * 5,
        "10v": torch.randn(1, 2, n_lat, n_lon) * 5,
        "msl": torch.ones(1, 2, n_lat, n_lon) * 101325,
    }
    
    current_atmos_vars = {
        "z": torch.randn(1, 2, n_levels, n_lat, n_lon) * 1000,
        "u": torch.randn(1, 2, n_levels, n_lat, n_lon) * 20,
        "v": torch.randn(1, 2, n_levels, n_lat, n_lon) * 20,
        "t": torch.randn(1, 2, n_levels, n_lat, n_lon) * 20 + 250,
        "q": torch.rand(1, 2, n_levels, n_lat, n_lon) * 0.02,
    }
    
    static_vars = {
        "lsm": torch.rand(n_lat, n_lon),
        "z": torch.randn(n_lat, n_lon) * 1000,
        "slt": torch.rand(n_lat, n_lon) * 7,
    }
    
    # Perform rollout
    current_time = datetime(2024, 6, 1, 12, 0)
    time_step = timedelta(hours=6)  # Aurora typically works with 6-hour steps
    
    print(f"Starting rollout from {current_time}")
    
    for step in range(n_forecast_steps):
        batch = Batch(
            surf_vars=current_surf_vars,
            static_vars=static_vars,
            atmos_vars=current_atmos_vars,
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=(current_time,),
                atmos_levels=(100, 250, 500, 850),
            ),
        )
        
        with torch.no_grad():
            prediction = model.forward(batch)
        
        predictions.append(prediction)
        
        # Update current state for next iteration
        # Use the prediction as the new initial condition
        # Keep the last two time steps for temporal context
        for var_name in current_surf_vars.keys():
            pred_value = prediction.surf_vars[var_name]
            current_surf_vars[var_name] = torch.cat([
                current_surf_vars[var_name][:, -1:, :, :],
                pred_value
            ], dim=1)
        
        for var_name in current_atmos_vars.keys():
            pred_value = prediction.atmos_vars[var_name]
            current_atmos_vars[var_name] = torch.cat([
                current_atmos_vars[var_name][:, -1:, :, :, :],
                pred_value
            ], dim=1)
        
        current_time += time_step
        
        if step % 3 == 0:
            temp_mean = prediction.surf_vars["2t"].mean().item()
            print(f"  Step {step+1}: {current_time} - Mean temp: {temp_mean:.2f} K")
    
    print(f"Completed {n_forecast_steps}-step forecast to {current_time}")
    print()
    
    return predictions, lat, lon


# ============================================================================
# Example 3: Physical Consistency Analysis
# ============================================================================
def example_physical_consistency():
    """
    Demonstrates how to analyze the physical consistency of Aurora predictions,
    such as conservation laws and physical relationships.
    """
    print("=" * 60)
    print("Example 3: Physical Consistency Analysis")
    print("=" * 60)
    
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()
    
    # Create physically consistent initial conditions
    n_lat, n_lon = 17, 32
    n_levels = 4
    
    # Temperature decreases with latitude (warmer at equator)
    lat = torch.linspace(90, -90, n_lat)
    lon = torch.linspace(0, 360, n_lon + 1)[:-1]
    lat_grid, _ = torch.meshgrid(lat, lon, indexing='ij')
    
    # Realistic temperature distribution
    base_temp = 288 - 30 * torch.abs(lat_grid / 90)  # Cooler at poles
    
    surf_vars = {
        "2t": base_temp.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "10u": torch.randn(1, 2, n_lat, n_lon) * 5,
        "10v": torch.randn(1, 2, n_lat, n_lon) * 5,
        "msl": torch.ones(1, 2, n_lat, n_lon) * 101325,
    }
    
    # Temperature decreases with height (pressure level)
    pressure_levels = torch.tensor([850, 500, 250, 100])  # hPa
    temp_profile = torch.zeros(1, 2, n_levels, n_lat, n_lon)
    
    for i, p_level in enumerate(pressure_levels):
        # Approximate temperature using standard atmosphere
        lapse_rate = 6.5e-3  # K/m
        height_approx = 44330 * (1 - (p_level / 1013.25) ** 0.1903)  # m
        temp_at_level = base_temp - lapse_rate * height_approx
        temp_profile[:, :, i, :, :] = temp_at_level.unsqueeze(0).unsqueeze(0)
    
    atmos_vars = {
        "z": torch.randn(1, 2, n_levels, n_lat, n_lon) * 1000,
        "u": torch.randn(1, 2, n_levels, n_lat, n_lon) * 20,
        "v": torch.randn(1, 2, n_levels, n_lat, n_lon) * 20,
        "t": temp_profile,
        "q": torch.rand(1, 2, n_levels, n_lat, n_lon) * 0.02,
    }
    
    static_vars = {
        "lsm": torch.rand(n_lat, n_lon),
        "z": torch.randn(n_lat, n_lon) * 1000,
        "slt": torch.rand(n_lat, n_lon) * 7,
    }
    
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2024, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )
    
    # Make prediction
    with torch.no_grad():
        prediction = model.forward(batch)
    
    # Analyze physical consistency
    print("Physical Consistency Checks:")
    print("-" * 40)
    
    # 1. Temperature range check
    pred_temp = prediction.surf_vars["2t"].squeeze()
    print(f"1. Temperature Range:")
    print(f"   Min: {pred_temp.min().item():.2f} K")
    print(f"   Max: {pred_temp.max().item():.2f} K")
    print(f"   Physically reasonable: {200 < pred_temp.min().item() < pred_temp.max().item() < 350}")
    
    # 2. Pressure range check
    pred_pressure = prediction.surf_vars["msl"].squeeze()
    print(f"\n2. Pressure Range:")
    print(f"   Min: {pred_pressure.min().item():.0f} Pa")
    print(f"   Max: {pred_pressure.max().item():.0f} Pa")
    print(f"   Physically reasonable: {90000 < pred_pressure.min().item() < pred_pressure.max().item() < 110000}")
    
    # 3. Wind speed check
    u_wind = prediction.surf_vars["10u"].squeeze()
    v_wind = prediction.surf_vars["10v"].squeeze()
    wind_speed = torch.sqrt(u_wind**2 + v_wind**2)
    print(f"\n3. Wind Speed:")
    print(f"   Mean: {wind_speed.mean().item():.2f} m/s")
    print(f"   Max:  {wind_speed.max().item():.2f} m/s")
    print(f"   Physically reasonable: {wind_speed.max().item() < 100}")
    
    # 4. Vertical temperature gradient (should generally decrease with height)
    atmos_temp = prediction.atmos_vars["t"].squeeze()
    temp_gradient = atmos_temp[0] - atmos_temp[-1]  # 850hPa - 100hPa
    print(f"\n4. Vertical Temperature Structure:")
    print(f"   Mean temp at 850 hPa: {atmos_temp[-1].mean().item():.2f} K")
    print(f"   Mean temp at 100 hPa: {atmos_temp[0].mean().item():.2f} K")
    print(f"   Temperature decreases with height: {temp_gradient.mean().item() > 0}")
    
    print()


# ============================================================================
# Example 4: Extreme Weather Event Simulation
# ============================================================================
def example_extreme_weather():
    """
    Demonstrates Aurora's capability to handle extreme weather conditions
    like tropical cyclones or heat waves.
    """
    print("=" * 60)
    print("Example 4: Extreme Weather Event Simulation")
    print("=" * 60)
    
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()
    
    n_lat, n_lon = 17, 32
    n_levels = 4
    
    lat = torch.linspace(90, -90, n_lat)
    lon = torch.linspace(0, 360, n_lon + 1)[:-1]
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
    
    # Create a cyclone-like initial condition
    # Center of the cyclone
    cyclone_lat = 15.0  # degrees
    cyclone_lon = 180.0  # degrees
    
    # Distance from cyclone center
    dist_from_center = torch.sqrt(
        (lat_grid - cyclone_lat)**2 + 
        (lon_grid - cyclone_lon)**2
    )
    
    # Pressure drops near the center (eye of the storm)
    pressure = 101325 - 3000 * torch.exp(-dist_from_center**2 / 100)
    
    # Temperature anomaly (warm core)
    temp_anomaly = 5 * torch.exp(-dist_from_center**2 / 200)
    temperature = 288 + temp_anomaly
    
    # Cyclonic wind pattern (counterclockwise in Northern Hemisphere)
    wind_speed = 20 * torch.exp(-dist_from_center**2 / 150)
    angle_to_center = torch.atan2(lat_grid - cyclone_lat, lon_grid - cyclone_lon)
    
    # Tangential wind components
    u_wind = -wind_speed * torch.sin(angle_to_center)
    v_wind = wind_speed * torch.cos(angle_to_center)
    
    # Add some inflow component
    u_wind -= 0.2 * wind_speed * torch.cos(angle_to_center)
    v_wind -= 0.2 * wind_speed * torch.sin(angle_to_center)
    
    surf_vars = {
        "2t": temperature.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "10u": u_wind.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "10v": v_wind.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "msl": pressure.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
    }
    
    # Create corresponding atmospheric structure
    atmos_vars = {
        "z": torch.randn(1, 2, n_levels, n_lat, n_lon) * 1000,
        "u": u_wind.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(1, 2, n_levels, 1, 1) * 
             torch.linspace(1.5, 0.5, n_levels).view(1, 1, n_levels, 1, 1),
        "v": v_wind.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(1, 2, n_levels, 1, 1) * 
             torch.linspace(1.5, 0.5, n_levels).view(1, 1, n_levels, 1, 1),
        "t": temperature.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(1, 2, n_levels, 1, 1) - 
             torch.linspace(0, 50, n_levels).view(1, 1, n_levels, 1, 1),
        "q": torch.rand(1, 2, n_levels, n_lat, n_lon) * 0.02,
    }
    
    static_vars = {
        "lsm": torch.zeros(n_lat, n_lon),  # All ocean
        "z": torch.zeros(n_lat, n_lon),
        "slt": torch.zeros(n_lat, n_lon),
    }
    
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2024, 9, 1, 0, 0),),  # Hurricane season
            atmos_levels=(100, 250, 500, 850),
        ),
    )
    
    print("Initial Cyclone Conditions:")
    print(f"  Min pressure: {pressure.min().item():.0f} Pa")
    print(f"  Max wind speed: {torch.sqrt(u_wind**2 + v_wind**2).max().item():.1f} m/s")
    print(f"  Warm core temp anomaly: {temp_anomaly.max().item():.1f} K")
    
    # Forecast the cyclone evolution
    n_steps = 8  # 48 hours with 6-hour steps
    
    print("\nForecasting cyclone evolution...")
    
    for step in range(n_steps):
        with torch.no_grad():
            prediction = model.forward(batch)
        
        # Update batch for next iteration
        for var_name in surf_vars.keys():
            surf_vars[var_name] = torch.cat([
                surf_vars[var_name][:, -1:, :, :],
                prediction.surf_vars[var_name]
            ], dim=1)
        
        for var_name in atmos_vars.keys():
            atmos_vars[var_name] = torch.cat([
                atmos_vars[var_name][:, -1:, :, :, :],
                prediction.atmos_vars[var_name]
            ], dim=1)
        
        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=(datetime(2024, 9, 1, 0, 0) + timedelta(hours=6*(step+1)),),
                atmos_levels=(100, 250, 500, 850),
            ),
        )
        
        # Analyze cyclone state
        pred_pressure = prediction.surf_vars["msl"].squeeze()
        pred_u = prediction.surf_vars["10u"].squeeze()
        pred_v = prediction.surf_vars["10v"].squeeze()
        pred_wind_speed = torch.sqrt(pred_u**2 + pred_v**2)
        
        if (step + 1) % 2 == 0:
            print(f"  T+{6*(step+1):2d}h: Min P = {pred_pressure.min().item():.0f} Pa, "
                  f"Max wind = {pred_wind_speed.max().item():.1f} m/s")
    
    print()


# ============================================================================
# Example 5: Energy and Momentum Analysis
# ============================================================================
def example_energy_analysis():
    """
    Analyzes energy and momentum conservation in Aurora predictions.
    """
    print("=" * 60)
    print("Example 5: Energy and Momentum Analysis")
    print("=" * 60)
    
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()
    
    n_lat, n_lon = 17, 32
    n_levels = 4
    
    # Create initial state
    lat = torch.linspace(90, -90, n_lat)
    lon = torch.linspace(0, 360, n_lon + 1)[:-1]
    
    surf_vars = {
        "2t": torch.ones(1, 2, n_lat, n_lon) * 288,
        "10u": torch.randn(1, 2, n_lat, n_lon) * 10,
        "10v": torch.randn(1, 2, n_lat, n_lon) * 10,
        "msl": torch.ones(1, 2, n_lat, n_lon) * 101325,
    }
    
    atmos_vars = {
        "z": torch.randn(1, 2, n_levels, n_lat, n_lon) * 1000,
        "u": torch.randn(1, 2, n_levels, n_lat, n_lon) * 20,
        "v": torch.randn(1, 2, n_levels, n_lat, n_lon) * 20,
        "t": torch.randn(1, 2, n_levels, n_lat, n_lon) * 10 + 270,
        "q": torch.rand(1, 2, n_levels, n_lat, n_lon) * 0.01,
    }
    
    static_vars = {
        "lsm": torch.rand(n_lat, n_lon),
        "z": torch.randn(n_lat, n_lon) * 500,
        "slt": torch.rand(n_lat, n_lon) * 7,
    }
    
    def calculate_energy(surf_vars, atmos_vars):
        """Calculate total energy in the system."""
        # Kinetic energy at surface
        ke_surf = 0.5 * (surf_vars["10u"]**2 + surf_vars["10v"]**2).mean()
        
        # Kinetic energy in atmosphere
        ke_atmos = 0.5 * (atmos_vars["u"]**2 + atmos_vars["v"]**2).mean()
        
        # Thermal energy (proportional to temperature)
        c_p = 1004.0  # Specific heat capacity of air J/(kg·K)
        te_surf = c_p * surf_vars["2t"].mean()
        te_atmos = c_p * atmos_vars["t"].mean()
        
        # Potential energy (simplified, using geopotential)
        pe = 9.81 * atmos_vars["z"].mean()
        
        return {
            "kinetic_surface": ke_surf.item(),
            "kinetic_atmos": ke_atmos.item(),
            "thermal_surface": te_surf.item(),
            "thermal_atmos": te_atmos.item(),
            "potential": pe.item()
        }
    
    # Calculate initial energy
    initial_energy = calculate_energy(surf_vars, atmos_vars)
    
    print("Initial Energy State:")
    for key, value in initial_energy.items():
        print(f"  {key:20s}: {value:12.2f}")
    
    # Run forecast
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2024, 6, 1, 0, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )
    
    with torch.no_grad():
        prediction = model.forward(batch)
    
    # Calculate final energy
    final_energy = calculate_energy(prediction.surf_vars, prediction.atmos_vars)
    
    print("\nFinal Energy State (after 6 hours):")
    for key, value in final_energy.items():
        print(f"  {key:20s}: {value:12.2f}")
    
    print("\nEnergy Changes:")
    for key in initial_energy.keys():
        change = final_energy[key] - initial_energy[key]
        pct_change = 100 * change / initial_energy[key] if initial_energy[key] != 0 else 0
        print(f"  {key:20s}: {change:+12.2f} ({pct_change:+.1f}%)")
    
    # Calculate momentum
    def calculate_momentum(surf_vars, atmos_vars):
        """Calculate total momentum in the system."""
        # Assuming unit density for simplification
        momentum_u_surf = surf_vars["10u"].mean()
        momentum_v_surf = surf_vars["10v"].mean()
        momentum_u_atmos = atmos_vars["u"].mean()
        momentum_v_atmos = atmos_vars["v"].mean()
        
        return {
            "u_surface": momentum_u_surf.item(),
            "v_surface": momentum_v_surf.item(),
            "u_atmos": momentum_u_atmos.item(),
            "v_atmos": momentum_v_atmos.item(),
        }
    
    initial_momentum = calculate_momentum(surf_vars, atmos_vars)
    final_momentum = calculate_momentum(prediction.surf_vars, prediction.atmos_vars)
    
    print("\nMomentum Conservation:")
    print("Initial Momentum:")
    for key, value in initial_momentum.items():
        print(f"  {key:12s}: {value:8.3f}")
    
    print("Final Momentum:")
    for key, value in final_momentum.items():
        print(f"  {key:12s}: {value:8.3f}")
    
    print()


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Aurora Tutorial - Scientific Machine Learning Examples")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_basic_weather_prediction()
    predictions, lat, lon = example_multistep_forecast()
    example_physical_consistency()
    example_extreme_weather()
    example_energy_analysis()
    
    print("=" * 60)
    print("Tutorial completed successfully!")
    print("=" * 60)
