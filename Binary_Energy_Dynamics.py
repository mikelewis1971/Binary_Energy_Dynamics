import math
import hashlib
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind


# ----------------------------
''' 1. Blockchain Implementation

The Purpose of the blockchain is to serve as a ledger and a way of tracking divide by zero events.  It serves as a 
psuedo-entanglement variable.  While not true entanglement, you can achieve reversibility via the blockchain.  
This Blockchain is not the real blockchain I recommend but it's a proof of concept.

The real blockchain only focuses on energy and ensures every transaction resolves to 0 so we know that all requests are legitimate and
accurate.  I will release that at some later date.  Or feel free to figure it out on your own.
'''
# ----------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        """
        Initializes a new block in the blockchain.

        Parameters:
            index (int): Position of the block in the chain.
            timestamp (str): Time of block creation.
            data (dict): Data contained within the block.
            previous_hash (str): Hash of the preceding block.
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data  # Contains mod and anti-mod values along with energy metrics
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Computes the SHA-256 hash of the block's contents.

        Returns:
            str: The computed hash in hexadecimal format.
        """
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        """
        Initializes the blockchain with the genesis block.
        """
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """
        Creates the genesis (first) block in the blockchain.
        """
        genesis_block = Block(0, str(datetime.datetime.now()), {"data": "Genesis Block"}, "0")
        self.chain.append(genesis_block)
        print(f"Genesis Block Created: {genesis_block.hash}")

    def add_block(self, data):
        """
        Adds a new block to the blockchain with the provided data.

        Parameters:
            data (dict): Data to be stored in the block.
        """
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), str(datetime.datetime.now()), data, previous_block.hash)
        self.chain.append(new_block)
        print(f"Block {new_block.index} Added: {new_block.hash}")

    def is_chain_valid(self):
        """
        Validates the integrity of the blockchain.

        Returns:
            bool: True if the chain is valid, False otherwise.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Recompute the hash and compare
            if current.hash != current.compute_hash():
                print(f"Block {i} has invalid hash.")
                return False

            if current.previous_hash != previous.hash:
                print(f"Block {i} has invalid previous hash.")
                return False

        print("Blockchain is valid.")
        return True


# ----------------------------
# 2. Primality Checking Function
# ----------------------------
def is_prime(n):
    """
    Checks if a number is prime based on factorization.

    Parameters:
        n (int): The number to check.

    Returns:
        bool: True if prime, False otherwise.
    """
    if n <= 1:
        return False
    factors = factorize(n)
    if len(factors) == 1 and list(factors.values())[0] == 1:
        return True
    return False


# ----------------------------
# 3. Chiral Shifts Generation
# ----------------------------
def generate_chiral_shifts(n, max_rotations=5):
    """
    Generates a list of chiral shifts by rotating the binary digits (bits) of the number
    from right to left.

    Parameters:
        n (int): The number to generate shifts for.
        max_rotations (int): Maximum number of rotations to perform.

    Returns:
        list: List of unique shifted numbers.
    """
    shifts = set()
    binary_str = bin(n)[2:]  # Remove '0b' prefix
    length = len(binary_str)

    # Invert the binary string to process from right to left
    inverted_binary = binary_str[::-1]
    print(f"\nGenerating chiral shifts for Number: {n} (Binary: {binary_str}, Inverted: {inverted_binary})")

    # Right rotations (equivalent to left rotations on the inverted binary)
    rotated = inverted_binary
    for _ in range(max_rotations):
        rotated = rotated[1:] + rotated[0]
        shifted_num = int(rotated[::-1], 2)  # Re-invert back to original order
        shifts.add(shifted_num)
        print(f"  Right Rotation: {rotated[::-1]} -> {shifted_num}")

    # Left rotations (equivalent to right rotations on the inverted binary)
    rotated = inverted_binary
    for _ in range(max_rotations):
        rotated = rotated[-1] + rotated[:-1]
        shifted_num = int(rotated[::-1], 2)  # Re-invert back to original order
        shifts.add(shifted_num)
        print(f"  Left Rotation: {rotated[::-1]} -> {shifted_num}")

    shifts.add(n)  # Include original number
    print(f"  Original Number Included: {n}")

    unique_shifts = sorted(list(shifts))
    print(f"  Unique Shifts for {n}: {unique_shifts}")
    return unique_shifts


# ----------------------------
# 4. Energy Sequence Calculation
# ----------------------------
def calculate_corrected_energy(binary_bits):
    """
    Calculates the energy sequence based on bit transitions from right to left.

    Parameters:
        binary_bits (list): List of binary bits (0s and 1s), ordered from right to left.

    Returns:
        list: Energy sequence excluding the leading 0.
    """
    energy_sequence = [0]  # Start with 0 energy
    previous_bit = 0  # Initial previous bit

    #print(f"    Calculating energy sequence for bits: {binary_bits}")

    for bit in binary_bits:
        n1 = previous_bit
        n2 = bit
        if n1 == 0 and n2 == 1:
            energy = 1
        elif n1 == 1 and n2 == 0:
            energy = -1
        else:
            energy = 0
        energy_sequence.append(energy)
        #print(f"      Transition {n1} -> {n2}: Energy = {energy}")
        previous_bit = n2

    print(f"    Energy Sequence: {energy_sequence}")
    return energy_sequence


# ----------------------------
# 5. Hamiltonian Calculation
# ----------------------------
def calculate_hamiltonian(energy_sequence):
    """
    Calculates the Hamiltonian of an energy sequence (Kinetic + Potential).

    Parameters:
        energy_sequence (list): List of energy values.

    Returns:
        dict: Dictionary containing kinetic, potential, and total Hamiltonian.
    """
    if not energy_sequence:
        return {"Kinetic Energy": 0, "Potential Energy": 0, "Total Hamiltonian": 0}
    kinetic_energy = sum((energy_sequence[i] - energy_sequence[i - 1]) ** 2 for i in range(1, len(energy_sequence)))
    potential_energy = sum(e ** 2 for e in energy_sequence)
    total_hamiltonian = kinetic_energy + potential_energy
    print(f"      Kinetic Energy: {kinetic_energy}")
    print(f"      Potential Energy: {potential_energy}")
    print(f"    Calculating Hamiltonian:")
    print(f"      Total Hamiltonian: {total_hamiltonian}")
    return {
        "Kinetic Energy": kinetic_energy,
        "Potential Energy": potential_energy,
        "Total Hamiltonian": total_hamiltonian
    }


# ----------------------------
# 6. Lagrangian Calculation
# ----------------------------
def calculate_lagrangian(energy_sequence):
    """
    Calculates the Lagrangian (Kinetic - Potential).

    Parameters:
        energy_sequence (list): List of energy values.

    Returns:
        dict: Dictionary containing kinetic, potential, and total Lagrangian.
    """
    if not energy_sequence:
        return {"Kinetic Energy": 0, "Potential Energy": 0, "Total Lagrangian": 0}
    kinetic_energy = sum((energy_sequence[i] - energy_sequence[i - 1]) ** 2 for i in range(1, len(energy_sequence)))
    potential_energy = sum(e ** 2 for e in energy_sequence)
    total_lagrangian = kinetic_energy - potential_energy
    print(f"    Calculating Lagrangian:")
    print(f"      Total Lagrangian: {total_lagrangian}")
    return {
        "Kinetic Energy": kinetic_energy,
        "Potential Energy": potential_energy,
        "Total Lagrangian": total_lagrangian
    }


# ----------------------------
# 7. Energy Sums and Products Calculation
# ----------------------------
def calculate_energy_sums_and_products(energy_sequence):
    """
    Calculates positive sum, negative sum, total sum, and energy product.

    Parameters:
        energy_sequence (list): List of energy values.

    Returns:
        dict: Dictionary containing positive sum, negative sum, total sum, and energy product.
    """
    positive_sum = sum(e for e in energy_sequence if e > 0)
    negative_sum = sum(e for e in energy_sequence if e < 0)
    total_sum = sum(energy_sequence)

    energy_product = 1
    for e in energy_sequence:
        energy_product *= e  # Preserve zero energies

    print(f"    Calculating Energy Sums and Products:")
    print(f"      Positive Sum: {positive_sum}")
    print(f"      Negative Sum: {negative_sum}")
    print(f"      Total Sum: {total_sum}")
    print(f"      Energy Product: {energy_product}")

    return {
        "Positive Sum": positive_sum,
        "Negative Sum": negative_sum,
        "Total Sum": total_sum,
        "Energy Product": energy_product
    }


# ----------------------------
# 8. Anti-Modular Analysis
# ----------------------------
def anti_mod_analysis(n, m):
    """
    Performs Anti-Modular Analysis on a number n with modulus m.

    Parameters:
        n (int): The number to analyze.
        m (int): The modulus value.

    Returns:
        dict: Dictionary containing residues, positive energy, negative energy, and total energy.
    """
    residues = []
    positive_energy = 0
    negative_energy = 0

    print(f"    Performing Anti-Modular Analysis with modulus {m}:")
    for i in range(m):
        residue = i % m
        residues.append(residue)
        if residue > 0:
            positive_energy += residue
        elif residue < 0:
            negative_energy += residue
        #print(f"      i={i}: residue={residue}")

    total_energy = positive_energy + negative_energy
    print(f"      Residues: {residues}")
    print(f"      Positive Energy: {positive_energy}")
    print(f"      Negative Energy: {negative_energy}")
    print(f"      Total Energy: {total_energy}")

    return {
        "Residues": residues,
        "Positive Energy": positive_energy,
        "Negative Energy": negative_energy,
        "Total Energy": total_energy,
    }


# ----------------------------
# 9. Factorization Function
# ----------------------------
def factorize(n):
    """
    Factorizes a number into its prime factors.

    Parameters:
        n (int): The number to factorize.

    Returns:
        dict: Dictionary of prime factors with their exponents.
    """
    factors = {}
    original_n = n
    print(f"      Factorizing {n}:")
    # Check divisibility by 2
    two = 0
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        two = factors[2]
        n = n // 2
    #print(f'2 ** {two}')
    # Check for odd factors
    divisor = 3
    max_divisor = math.isqrt(n) + 1
    while divisor <= max_divisor and n > 1:
        while n % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            #print(f"       Factor: {divisor}, Exponent: {factors[divisor]}")
            n = n // divisor
            max_divisor = math.isqrt(n) + 1
        divisor += 2
    # If n is a prime number greater than 2
    if n > 1:
        factors[n] = 1
        #print(f"    **    Factor: {n}, Exponent: 1")
    # If no factors found, n is prime
    if not factors:
        factors[original_n] = 1
        #print(f"        Factor: {original_n}, Exponent: 1")
    print(factors)
    return factors


# ----------------------------
# 10. Analyze Single Number Function
# ----------------------------
def analyze_number(n, blockchain):
    """
    Analyzes a number to compute all energy dynamics and factorization, and logs data to the blockchain.

    Parameters:
        n (int): The number to analyze.
        blockchain (Blockchain): The blockchain ledger to log data.

    Returns:
        dict: Dictionary containing all analysis results.
    """
    print(f"\nAnalyzing Number: {n}")
    analysis = {}
    is_n_prime = is_prime(n)
    analysis["Is_Prime"] = is_n_prime
    print(f"  Is Prime: {is_n_prime}")

    # Generate chiral shifts
    shifts = generate_chiral_shifts(n)
    analysis["Chiral_Shifts"] = shifts

    # Initialize dictionaries to hold analysis for each shift
    shift_analyses = {}

    for shift in shifts:
        print(f"\n  Analyzing Shift: {shift}")
        shift_analysis = {}

        # Binary representation (right to left)
        binary_str = bin(shift)[2:]
        binary_bits = [int(bit) for bit in binary_str[::-1]]  # Invert to process right to left
        shift_analysis["Binary"] = binary_bits
        print(f"    Binary Representation (right to left): {binary_bits}")

        # Calculate energy sequence
        energy_sequence = calculate_corrected_energy(binary_bits)
        shift_analysis["Energy_Sequence"] = energy_sequence

        # Calculate Hamiltonian
        hamiltonian = calculate_hamiltonian(energy_sequence)
        shift_analysis["Hamiltonian"] = hamiltonian

        # Calculate Lagrangian
        lagrangian = calculate_lagrangian(energy_sequence)
        shift_analysis["Lagrangian"] = lagrangian

        # Calculate Energy Sums and Products
        energy_sums_products = calculate_energy_sums_and_products(energy_sequence)
        shift_analysis["Energy_Sums_Products"] = energy_sums_products

        # Anti-Modular Analysis (using modulus as length of binary sequence)
        m = len(binary_bits)
        anti_mod = anti_mod_analysis(shift, m)
        shift_analysis["Anti_Modular"] = anti_mod

        # Factorization
        factors = factorize(shift)
        shift_analysis["Factors"] = factors

        # Vector Representation for Banach Spaces
        energy_vector = np.array(energy_sequence)
        norm = np.linalg.norm(energy_vector)  # Euclidean norm
        shift_analysis["Energy_Vector"] = energy_vector
        shift_analysis["Vector_Norm"] = norm
        #print(f"    Energy Vector: {energy_vector}")
        print(f"    Vector Norm (Euclidean): {norm:.4f}")

        # Routhian Calculation (Example Implementation)
        # Routhian = Kinetic Energy - Potential Energy + Constraint Force (Q)
        # For simplicity, assume Q = 0 in this context
        routhian = hamiltonian["Kinetic Energy"] - hamiltonian["Potential Energy"]  # + Q (assuming Q=0)
        shift_analysis["Routhian"] = routhian
        print(f"    Routhian: {routhian}")

        # Store shift analysis
        shift_analyses[shift] = shift_analysis

        # Prepare data for blockchain logging
        block_data = {
            "Number": n,
            "Shift": shift,
            "Is_Prime": is_prime(shift),
            "Binary": ''.join(map(str, binary_bits[::-1])),  # Re-invert to original binary order for storage
            "Energy_Sequence": energy_sequence,
            "Hamiltonian": hamiltonian["Total Hamiltonian"],
            "Lagrangian": lagrangian["Total Lagrangian"],
            "Positive_Sum": energy_sums_products["Positive Sum"],
            "Negative_Sum": energy_sums_products["Negative Sum"],
            "Total_Sum": energy_sums_products["Total Sum"],
            "Energy_Product": energy_sums_products["Energy Product"],
            "Anti_Mod_Total_Energy": anti_mod["Total Energy"],
            "Energy_Vector": energy_vector.tolist(),  # Convert NumPy array to list for JSON serialization
            "Vector_Norm": norm,
            "Routhian": routhian,
            "Factors": factors
        }

        # Add block to blockchain
        blockchain.add_block(block_data)

    analysis["Shifts_Analysis"] = shift_analyses
    return analysis


# ----------------------------
# 11. Collect Energy Metrics Function
# ----------------------------
def collect_energy_metrics(numbers, blockchain):
    """
    Collects energy metrics for a list of numbers and logs them to the blockchain.

    Parameters:
        numbers (list): List of integers to analyze.
        blockchain (Blockchain): The blockchain ledger to log data.

    Returns:
        pandas.DataFrame: DataFrame containing energy metrics and primality status.
    """
    data = []
    total_numbers = len(numbers)
    for idx, num in enumerate(numbers, start=1):
        print(f"\n=== Processing {idx}/{total_numbers}: Number {num} ===")
        analysis = analyze_number(num, blockchain)
        print('*')
        # Aggregate data for the original number (n)
        original_shift = num
        shift_analysis = analysis["Shifts_Analysis"][original_shift]

        # Convert Energy Vector to feature for Banach Space analysis
        energy_vector = np.array(shift_analysis["Energy_Vector"])
        norm = shift_analysis["Vector_Norm"]
        routhian = shift_analysis["Routhian"]

        metrics = {
            "Number": num,
            "Is_Prime": analysis["Is_Prime"],
            "Hamiltonian": shift_analysis["Hamiltonian"]["Total Hamiltonian"],
            "Lagrangian": shift_analysis["Lagrangian"]["Total Lagrangian"],
            "Positive_Sum": shift_analysis["Energy_Sums_Products"]["Positive Sum"],
            "Negative_Sum": shift_analysis["Energy_Sums_Products"]["Negative Sum"],
            "Total_Sum": shift_analysis["Energy_Sums_Products"]["Total Sum"],
            "Energy_Product": shift_analysis["Energy_Sums_Products"]["Energy Product"],
            "Total_Energy": shift_analysis["Anti_Modular"]["Total Energy"],
            "Energy_Vector_Norm": norm,
            "Routhian": routhian,
            "Is_Mersenne_Prime": False,  # To be updated later
            "Is_Fermat_Prime": False  # To be updated later
        }
        data.append(metrics)
    df = pd.DataFrame(data)
    return df


# ----------------------------
# 12. Statistical Comparison Function
# ----------------------------
def statistical_comparison(df):
    """
    Performs statistical comparisons between primes and composites for Hamiltonian and Lagrangian.

    Parameters:
        df (pandas.DataFrame): DataFrame containing energy metrics and primality status.

    Returns:
        None: Prints the results of t-tests.
    """
    primes_df = df[df['Is_Prime'] == True]
    composites_df = df[df['Is_Prime'] == False]

    print("\n--- Statistical Comparison: Primes vs Composites ---")
    print(f"Number of Primes: {len(primes_df)}")
    print(f"Number of Composites: {len(composites_df)}")

    # T-test for Hamiltonian
    if len(primes_df) > 1 and len(composites_df) > 1:
        t_stat_ham, p_val_ham = ttest_ind(primes_df['Hamiltonian'], composites_df['Hamiltonian'], equal_var=False)
        print("\nT-test for Hamiltonian between Primes and Composites:")
        print(f"  T-statistic: {t_stat_ham:.4f}, P-value: {p_val_ham:.4f}")

        # T-test for Lagrangian
        t_stat_lag, p_val_lag = ttest_ind(primes_df['Lagrangian'], composites_df['Lagrangian'], equal_var=False)
        print("\nT-test for Lagrangian between Primes and Composites:")
        print(f"  T-statistic: {t_stat_lag:.4f}, P-value: {p_val_lag:.4f}")
    else:
        print("\nInsufficient data for statistical comparison.")

    # Interpretation
    alpha = 0.05
    if 'p_val_ham' in locals():
        if p_val_ham < alpha:
            print("\nHamiltonian: Significant difference between primes and composites.")
        else:
            print("\nHamiltonian: No significant difference between primes and composites.")

        if p_val_lag < alpha:
            print("Lagrangian: Significant difference between primes and composites.")
        else:
            print("Lagrangian: No significant difference between primes and composites.")
    else:
        print("\nStatistical comparison skipped due to insufficient data.")


# ----------------------------
# 13. Correlation Function
# ----------------------------
def correlation_total_sum_primality(df):
    """
    Calculates the correlation between Total_Sum and Is_Prime.

    Parameters:
        df (pandas.DataFrame): DataFrame containing energy metrics and primality status.

    Returns:
        None: Prints the correlation coefficient.
    """
    if 'Is_Prime' in df.columns and 'Total_Sum' in df.columns:
        correlation = df['Total_Sum'].corr(df['Is_Prime'].astype(int))
        print(f"\nCorrelation between Total_Sum and Is_Prime: {correlation:.4f}")
    else:
        print("\nCorrelation analysis skipped due to missing columns.")


# ----------------------------
# 14. Analyze Large Dataset Function
# ----------------------------
def sieve_of_eratosthenes(limit):
    """
    Generates a list of prime numbers up to the specified limit using the Sieve of Eratosthenes.

    Parameters:
        limit (int): Upper bound for generating primes.

    Returns:
        list: List of prime numbers up to 'limit'.
    """
    sieve = [True] * (limit + 1)
    sieve[0:2] = [False, False]
    for current in range(2, int(math.isqrt(limit)) + 1):
        if sieve[current]:
            sieve[current * current: limit + 1: current] = [False] * len(range(current * current, limit + 1, current))
    return [num for num, is_p in enumerate(sieve) if is_p]


def analyze_large_dataset():
    """
    Analyzes a large dataset of numbers, including primes, composites, Mersenne primes, and Fermat primes.

    Returns:
        pandas.DataFrame: DataFrame containing energy metrics and primality status.
        Blockchain: The blockchain ledger containing all analyses.
    """
    print("\n=== Starting Large Dataset Analysis ===")
    # Initialize blockchain
    blockchain = Blockchain()

    # Define the range
    limit = 1000
    #primes = sieve_of_eratosthenes(limit)
    #composites = list(set(range(2, limit + 1)) - set(primes))

    # Define known Mersenne and Fermat primes within the range
    #mersenne_primes = [3, 7, 31, 127]  # 8191 is beyond 1000, excluded
    #fermat_primes = [3, 5, 17, 257]  # 65537 is beyond 1000, excluded

    # Filter Mersenne and Fermat primes within the limit
    #mersenne_primes = [p for p in mersenne_primes if p <= limit]
    #fermat_primes = [p for p in fermat_primes if p <= limit]

    #print(f"\nNumber of Primes: {len(primes)}")
    #print(f"Number of Composites: {len(composites)}")
    #print(f"Identified Mersenne Primes within {limit}: {mersenne_primes}")
    #print(f"Identified Fermat Primes within {limit}: {fermat_primes}")

    # Combine into a unique set
    #test_numbers = sorted(list(set(primes + composites + mersenne_primes + fermat_primes)))
    test_numbers = [8678787897979879878]
    # Perform analysis and log to blockchain
    df = collect_energy_metrics(test_numbers, blockchain)

    # Identify Mersenne and Fermat primes
    #df['Is_Mersenne_Prime'] = df['Number'].isin(mersenne_primes)
    #df['Is_Fermat_Prime'] = df['Number'].isin(fermat_primes)

    # Save the DataFrame for reference
    df.to_csv("energy_metrics_large_dataset_with_special_primes.csv", index=False)
    print(
        "\nLarge dataset analysis completed. Results saved to 'energy_metrics_large_dataset_with_special_primes.csv'.")

    return df, blockchain


# ----------------------------
# 15. Main Execution Function
# ----------------------------
def main():
    """
    Main function to execute the Enhanced Binary Energy Dynamics Test Suite with Blockchain Integration.
    """
    print("=== Binary Energy Dynamics Test Suite with Blockchain Integration ===")

    # Analyze large dataset
    df_large, blockchain = analyze_large_dataset()

    # Perform Statistical Comparison
    statistical_comparison(df_large)

    # Perform Correlation Analysis
    correlation_total_sum_primality(df_large)
'''
    # Visualization: Correlation Heatmap
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_large.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Heatmap of Energy Metrics (Numbers 2-1000)')
    plt.show()

    # Visualization: Pairplot
    print("\nGenerating Pairplot...")
    try:
        sns.pairplot(df_large, hue='Is_Prime', palette={True: 'green', False: 'red'},
                     vars=['Hamiltonian', 'Lagrangian', 'Total_Sum', 'Energy_Product'])
        plt.suptitle('Pairplot of Energy Metrics Colored by Primality', y=1.02)
        plt.show()
    except KeyError as e:
        print(f"Pairplot Error: {e}. Ensure that 'Is_Prime' column contains boolean values.")

    # Visualization: Hamiltonian vs Lagrangian
    print("\nGenerating Scatter Plot: Hamiltonian vs Lagrangian...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Hamiltonian', y='Lagrangian', hue='Is_Prime', data=df_large,
                    palette={True: 'green', False: 'red'}, alpha=0.6)
    plt.title('Hamiltonian vs Lagrangian Colored by Primality (Numbers 2-1000)')
    plt.show()
'''


# ----------------------------
# 16. Run the Main Function
# ----------------------------
if __name__ == "__main__":
    main()
    print("\n=== Test Suite Execution Completed ===")

