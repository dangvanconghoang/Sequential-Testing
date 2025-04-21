# === Configuration ===
MAX_CONVERSIONS = 800_000
MAX_BARRIER = 5000

def log_beta(a, b):
    """Compute the natural logarithm of the Beta function."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

# === Core Sequential Estimator Functions ===

def compute_required_sample_size(z, alpha, power, null_p, alt_p, max_conversions=MAX_CONVERSIONS):
    """
    Given a Z-barrier, find the smallest sample size n such that:
    - P_alt > power
    - P_null < alpha
    """
    null_cdf = 0.0
    alt_cdf = 0.0

    log_null_p = math.log(null_p)
    log_null_1_p = math.log(1.0 - null_p)
    log_alt_p = math.log(alt_p)
    log_alt_1_p = math.log(1.0 - alt_p)

    for n in range(z, max_conversions + 1, 2):
        k = 0.5 * (n + z)
        weight = z / n / k
        lbeta_val = log_beta(k, n + 1 - k)

        null_cdf += weight * math.exp(-lbeta_val + (k - z) * log_null_p + k * log_null_1_p)
        alt_cdf += weight * math.exp(-lbeta_val + (k - z) * log_alt_p + k * log_alt_1_p)

        if math.isnan(null_cdf) or math.isnan(alt_cdf):
            return np.nan

        if alt_cdf > power:
            if null_cdf < alpha:
                return n
            else:
                return np.nan

    return np.nan

def search_optimal_z(z_min, z_max, alpha, power, null_p, alt_p, max_conversions=MAX_CONVERSIONS):
    """
    Binary search for Z-barrier such that:
    - under H0: cumulative probability < alpha
    - under H1: cumulative probability > power
    """
    log_null_p = math.log(null_p)
    log_null_1_p = math.log(1.0 - null_p)
    log_alt_p = math.log(alt_p)
    log_alt_1_p = math.log(1.0 - alt_p)

    z = z_min + 2 * ((z_max - z_min) // 4)

    while z_min < z_max:
        null_cdf = 0.0
        alt_cdf = 0.0

        # Compute cumulative probabilities for null and alternative hypotheses
        for n in range(z, max_conversions + 1, 2):
            k = 0.5 * (n + z)
            weight = z / n / k
            lbeta_val = log_beta(k, n + 1 - k)

            null_cdf += weight * math.exp(-lbeta_val + (k - z) * log_null_p + k * log_null_1_p)
            alt_cdf += weight * math.exp(-lbeta_val + (k - z) * log_alt_p + k * log_alt_1_p)

            if math.isnan(null_cdf) or math.isnan(alt_cdf):
                break

            # Check if both constraints are satisfied
            if alt_cdf > power:
                if null_cdf < alpha:
                    z_max = z
                else:
                    z_min = z + 2
                break
            elif null_cdf > alpha:
                z_min = z + 2
                break

        if math.isnan(null_cdf) or math.isnan(alt_cdf) or n >= max_conversions:
            break

        z = z_min + 2 * ((z_max - z_min) // 4)

    return z, z_max

def estimate_sequential_sample_size(alpha, power, baseline_rate, effect_size):
    """
    Estimate the minimum sample size per group for a sequential test.
    - baseline_rate: control group conversion rate
    - effect_size: absolute minimal detectable effect (delta)
    """
    null_p = 0.5 # Assumed 50/50 prior odds
    alt_p = 1.0 / (1.0 + (baseline_rate + effect_size) / baseline_rate) # Implied odds under alternative hypothesis

    # Search for Z-barrier for both odd and even values
    best_odd_z, _ = search_optimal_z(1, MAX_BARRIER - 1, alpha, power, null_p, alt_p)
    best_even_z, _ = search_optimal_z(2, MAX_BARRIER, alpha, power, null_p, alt_p)

    odd_n = compute_required_sample_size(best_odd_z, alpha, power, null_p, alt_p)
    even_n = compute_required_sample_size(best_even_z, alpha, power, null_p, alt_p)

    # Choose the Z-barrier that results in the smaller sample size
    if math.isnan(odd_n) or (even_n < odd_n):
        return even_n, best_even_z
    else:
        return odd_n, best_odd_z


alpha = 0.05
power = 0.80
baseline_rate = 0.20           # 20% baseline
mde_relative = 0.10            # 10% relative MDE
effect_size = baseline_rate * mde_relative  # 2% absolute difference

sample_size, z_barrier = estimate_sequential_sample_size(alpha, power, baseline_rate, effect_size)

if math.isnan(sample_size):
    print("Unable to compute sample size — try increasing MDE or max conversions.")
else:
    print(f"Sample size per group: {sample_size}")
    print(f"Z-barrier: {z_barrier}")
