import jax
import jax.numpy as jnp
from jax import jit, lax

def compute_cost_matrix(x, y):
    """
    Computes the squared Euclidean cost matrix between two 1D point clouds.
    """
    return (x[:, None] - y[None, :]) ** 2

@jit
def sinkhorn_knopp(a, b, C, epsilon=1e-2, num_iters=100):
    """
    Solves the entropically regularized optimal transport problem using the Sinkhorn-Knopp algorithm.
    This runs very fast on GPU/TPU thanks to JAX.
    
    Args:
        a: Source distribution (shape: [n])
        b: Target distribution (shape: [m])
        C: Cost matrix (shape: [n, m])
        epsilon: Entropic regularization parameter (smaller = closer to exact OT, but harder to converge)
        num_iters: Number of Sinkhorn iterations
        
    Returns:
        P: Optimal transport plan (shape: [n, m])
    """
    # Gibbs kernel
    K = jnp.exp(-C / epsilon)
    
    # Initialize scaling vectors
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)

    # Sinkhorn iterations (using lax.scan for unrolling loops securely in XLA)
    def iteration(carry, _):
        u, v = carry
        u = a / jnp.dot(K, v)
        v = b / jnp.dot(K.T, u)
        return (u, v), None

    (u, v), _ = lax.scan(iteration, (u, v), None, length=num_iters)

    # Compute the final transport plan (coupling matrix)
    P = u[:, None] * K * v[None, :]
    return P

def main():
    print("Initializing Optimal Transport with JAX...")
    
    # Example: Transporting mass between two 1D uniform distributions
    n, m = 10, 15  # Number of points in source and target
    
    # Distributions (uniform probability mass)
    a = jnp.ones(n) / n
    b = jnp.ones(m) / m
    
    # Coordinates for the point clouds
    x = jnp.linspace(0, 1, n)
    y = jnp.linspace(0, 1, m)
    
    print(f"Source points (x): {n}")
    print(f"Target points (y): {m}")

    # 1. Compute Cost Matrix
    C = compute_cost_matrix(x, y)
    
    # 2. Run Sinkhorn-Knopp Algorithm
    epsilon = 0.01
    P = sinkhorn_knopp(a, b, C, epsilon=epsilon, num_iters=200)
    
    # 3. Compute the approximations
    transport_cost = jnp.sum(P * C)
    
    print("\n--- Results ---")
    print(f"Calculated Optimal Transport Cost: {transport_cost:.4f}")
    print(f"Coupling Matrix Shape: {P.shape}")
    
    # Verify mass conservation (marginal constraints)
    marginal_a = jnp.sum(P, axis=1)
    marginal_b = jnp.sum(P, axis=0)
    
    print("\nVerifying Constraints:")
    print(f"Source marginals match 'a': {jnp.allclose(marginal_a, a, atol=1e-3)}")
    print(f"Target marginals match 'b': {jnp.allclose(marginal_b, b, atol=1e-3)}")

if __name__ == "__main__":
    main()
