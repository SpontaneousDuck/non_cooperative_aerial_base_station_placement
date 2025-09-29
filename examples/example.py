"""Example usage of the UAV placement optimization package."""

from uav_placement import UAVPlacementOptimizer

def main():
    """Run a simple example optimization."""
    print("UAV Placement Optimization Example")
    print("==================================")
    
    # Create optimizer
    optimizer = UAVPlacementOptimizer()
    
    # Run optimization with custom parameters
    print("Running optimization...")
    results = optimizer.optimize(
        num_base_stations=3,
        num_mobile_users=30,
        region_size=7.0,
        bs_powers_dbm=[7, 9, 12],  # Different power levels
        num_iterations=50,
        optimization_method='stochastic',
        utility_function='sigmoid',
        random_seed=42  # For reproducibility
    )
    
    # Print results
    print(f"\nOptimization completed!")
    print(f"Method: {results['optimization_method']}")
    print(f"Final utility: {results['results']['final_utility']:.4f}")
    print(f"Converged: {results['results']['converged']}")
    
    print("\nOptimal base station positions:")
    for bs_id, pos in results['results']['final_bs_positions'].items():
        print(f"  {bs_id}: ({pos['x_km']:.3f}, {pos['y_km']:.3f}) km, "
              f"Power: {pos['power_dbm']:.1f} dBm")
    
    # Save results
    optimizer.save_results('optimization_results.json')
    print("\nResults saved to 'optimization_results.json'")
    
    # Create and save plot
    try:
        fig = optimizer.plot_results(save_path='optimization_plot.png')
        print("Plot saved to 'optimization_plot.png'")
    except ImportError:
        print("Matplotlib not available, skipping plot")
    
    # Quick comparison with k-means
    print("\nComparing with k-means optimization...")
    results_kmeans = UAVPlacementOptimizer.quick_optimize(
        num_base_stations=3,
        num_mobile_users=30,
        region_size=7.0,
        num_iterations=50
    )
    
    print(f"Stochastic final utility: {results['results']['final_utility']:.4f}")
    print(f"K-means final utility: {results_kmeans['results']['final_utility']:.4f}")

if __name__ == '__main__':
    main()
