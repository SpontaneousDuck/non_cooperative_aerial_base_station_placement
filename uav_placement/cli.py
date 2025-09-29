"""Command-line interface for UAV placement optimization."""

import argparse
import json
import sys
from pathlib import Path

from .optimization.optimizer import UAVPlacementOptimizer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Non-cooperative Aerial Base Station Placement Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument('--num-bs', type=int, default=3,
                      help='Number of aerial base stations')
    parser.add_argument('--num-users', type=int, default=50,
                      help='Number of mobile users')
    parser.add_argument('--region-size', type=float, default=7.0,
                      help='Size of the square region (km)')
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of optimization iterations')
    
    # Optimization parameters
    parser.add_argument('--method', choices=['stochastic', 'kmeans'], default='stochastic',
                      help='Optimization method')
    parser.add_argument('--utility', choices=['sigmoid', 'leakyRelu'], default='sigmoid',
                      help='Utility function type')
    parser.add_argument('--step-size', type=float, default=2.0,
                      help='Step size for gradient descent')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Mini-batch size (None for full batch)')
    
    # Power parameters
    parser.add_argument('--bs-powers', nargs='+', type=float, default=None,
                      help='Base station transmission powers in dBm')
    parser.add_argument('--power-threshold', type=float, default=-91.0,
                      help='Lower power threshold in dBm')
    parser.add_argument('--power-upper', type=float, default=-89.0,
                      help='Upper power threshold in dBm (for sigmoid)')
    
    # System parameters
    parser.add_argument('--bs-height', type=float, default=0.03,
                      help='Base station height in km')
    parser.add_argument('--frequency', type=float, default=2.39,
                      help='Carrier frequency in GHz')
    
    # Output parameters
    parser.add_argument('--output', type=str, default=None,
                      help='Output file for results (JSON format)')
    parser.add_argument('--plot', type=str, default=None,
                      help='Save plot to file (PNG/PDF format)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--quiet', action='store_true',
                      help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.bs_powers and len(args.bs_powers) != args.num_bs:
        print(f"Error: Number of BS powers ({len(args.bs_powers)}) must match number of BSs ({args.num_bs})")
        sys.exit(1)
    
    try:
        # Create optimizer
        optimizer = UAVPlacementOptimizer()
        
        # Run optimization
        if not args.quiet:
            print("Starting UAV placement optimization...")
            print(f"Configuration: {args.num_bs} BSs, {args.num_users} users, {args.region_size} km region")
        
        results = optimizer.optimize(
            num_base_stations=args.num_bs,
            num_mobile_users=args.num_users,
            region_size=args.region_size,
            bs_powers_dbm=args.bs_powers,
            bs_height=args.bs_height,
            optimization_method=args.method,
            utility_function=args.utility,
            power_threshold_dbm=args.power_threshold,
            power_upper_dbm=args.power_upper,
            num_iterations=args.iterations,
            mini_batch_size=args.batch_size,
            step_size=args.step_size,
            frequency_ghz=args.frequency,
            random_seed=args.seed
        )
        
        if not args.quiet:
            print("\nOptimization completed!")
            print(f"Final utility: {results['results']['final_utility']:.4f}")
            print(f"Converged: {results['results']['converged']}")
            print(f"Utility improvement: {results['results']['utility_improvement']:.4f}")
        
        # Print final positions
        print("\nOptimal base station positions:")
        for bs_id, pos in results['results']['final_bs_positions'].items():
            print(f"  {bs_id}: ({pos['x_km']:.3f}, {pos['y_km']:.3f}, {pos['z_km']:.3f}) km, "
                  f"Power: {pos['power_dbm']:.1f} dBm")
        
        # Save results
        if args.output:
            optimizer.save_results(args.output)
            if not args.quiet:
                print(f"Results saved to {args.output}")
        
        # Save plot
        if args.plot:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                fig = optimizer.plot_results(save_path=args.plot)
                if not args.quiet:
                    print(f"Plot saved to {args.plot}")
            except ImportError:
                print("Warning: matplotlib not available, skipping plot")
        
        # Output results to stdout for programmatic access
        if args.quiet:
            print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()