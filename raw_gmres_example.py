import functools

import jax
import jax.numpy as jnp
import lineax as lx
import logging

jax.config.update("jax_log_compiles", True)
jax.config.update("jax_explain_cache_misses", True)

@jax.jit
def matvec(A_data, x):
    # A module-level function, closed over nothing -- reusable and testable on its
    # own, independent of any particular `solve` call.
    return A_data @ x


def main():
    n = 100
    solver = lx.GMRES(rtol=1e-6, atol=1e-6, restart=20)

    trace_count = 0

    def solve(A_data, b):
        # Counting here works because this function body (including the
        # `functools.partial` below) only actually runs while JAX is *tracing* -- a
        # cache hit on a later call with new data (same shape/dtype) skips straight
        # to the compiled XLA executable and never re-executes this Python code.
        nonlocal trace_count
        trace_count += 1

        mv = functools.partial(matvec, A_data)
        solution, result, stats = lx.raw_linear_solve(mv, b, solver)
        return solution, result, stats

    solve_jit = jax.jit(solve)

    key = jax.random.PRNGKey(0)
    for i in range(5):
        key, a_key, b_key = jax.random.split(key, 3)
        # A fresh matrix (and hence a fresh `functools.partial`) every iteration --
        # same shape/dtype each time, so no retracing should occur after the first
        # call.
        A_data = jnp.eye(n) + 0.01 * jax.random.normal(a_key, (n, n))
        b = jax.random.normal(b_key, (n,))

        solution, result, stats = solve_jit(A_data, b)
        residual = jnp.linalg.norm(A_data @ solution - b)

        logging.info(
            f"iter {i}: result={result}, num_steps={stats['num_steps']}, "
            f"residual={residual:.2e}, trace_count={trace_count}"
        )

    assert trace_count == 1, (
        f"expected exactly 1 trace across all 5 calls, got {trace_count}"
    )
    logging.info("No retracing occurred across repeated solves with new data.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:lineax_example: %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    main()
