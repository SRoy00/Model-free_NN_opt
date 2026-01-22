import numpy as np
from scipy.fft import fft, ifft, fftfreq
import functools
from jax.scipy.linalg import sqrtm
import jax.numpy as jnp
from jax import config, jit
import jax
import dynamiqs as dq

config.update("jax_enable_x64", True)

def flattop_gaussian_pulse(input_array, amplitude, rise_fraction=0.05):
    """
    Modulate an input pulse shape with a flat-top Gaussian envelope.

    Parameters:
        input_array (jnp.ndarray): Input pulse shape array (can have complex values).
        amplitude (float): Maximum amplitude of the modulated pulse.
        rise_fraction (float): Fraction of the total pulse length used for the Gaussian rise/fall (default: 0.05).

    Returns:
        jnp.ndarray: Array representing the modulated flat-top Gaussian pulse.
    """
    num_points = len(input_array)

    if rise_fraction <= 0 or rise_fraction >= 0.5:
        raise ValueError("rise_fraction must be between 0 and 0.5.")

    # Determine the number of points for the rise and flat regions
    # rise_points = int(num_points * rise_fraction)
    rise_points = int(250*0.05)
    flat_points = num_points - 2 * rise_points

    if rise_points < 1:
        raise ValueError("The number of points for the rise regions must be at least 1.")
    if flat_points < 1:
        raise ValueError("The number of points for the flat region must be at least 1.")

    # Generate the Gaussian rise and fall portions
    sigma = rise_points / 5.0
    t_rise = jnp.linspace(-5 * sigma, 0, rise_points)
    gaussian_rise = jnp.exp(-(t_rise**2) / (2 * sigma**2))

    t_fall = jnp.linspace(0, 5 * sigma, rise_points)
    gaussian_fall = jnp.exp(-(t_fall**2) / (2 * sigma**2))

    # Generate the flat-top portion
    flat_top = jnp.ones(flat_points)

    # Concatenate rise, flat, and fall portions to form the envelope
    envelope = jnp.concatenate([gaussian_rise, flat_top, gaussian_fall])

    # Clip the real and imaginary parts of the input array separately
    real_part = jnp.clip(input_array.real, -amplitude, amplitude)
    imag_part = jnp.clip(input_array.imag, -amplitude, amplitude)

    # Recombine the clipped real and imaginary parts
    clipped_input_array = real_part + 1j * imag_part

    # Modulate the input array with the envelope
    modulated_pulse = clipped_input_array * envelope

    return modulated_pulse

def low_pass_filter(signal, max_freq, time_step):
    """
    Apply a low-pass filter to the input signal to remove frequency components higher than max_freq.

    Parameters:
    signal (np.ndarray): Input signal in the time domain (can be complex).
    max_freq (float): Maximum allowed frequency in MHz.
    time_step (float): Time step between consecutive points in seconds.

    Returns:
    np.ndarray: Filtered signal in the time domain (complex).
    """
    # Adjust the unit of max frequency to Hz
    max_freq *= 1e6

    # Perform FFT to get the frequency domain representation
    N = len(signal)
    freqs = fftfreq(N, time_step)
    signal_fft = fft(signal)

    # Apply the low-pass filter
    filtered_fft = signal_fft.copy()
    filtered_fft[np.abs(freqs) > max_freq] = 0

    # Perform inverse FFT to get the filtered signal in the time domain
    filtered_signal = ifft(filtered_fft)

    return filtered_signal

def project_amps(amps, max_freq, time_step = 4e-9):
     
    """
    Project the amplitudes to ensure no beyond the limits frequency component in the pulse.

    Parameters:
        amps (jnp.ndarray): Input array of amplitudes (can have complex values).
        max_freq (float): Maximum allowed amplitude difference corresponding to frequency difference between consecutive elements.

    Returns:
        jnp.ndarray: Array of projected amplitudes with differences within the specified range.
    """
     
    def apply_low_pass_filter(row):
        # Convert JAX array to NumPy array
        row_np = np.array(row, dtype=np.complex64)
        # Apply the low-pass filter
        filtered_row = low_pass_filter(row_np, max_freq, time_step)
        # Convert the filtered row back to JAX array
        return jnp.array(filtered_row)

    # Apply the low-pass filter to each row separately
    projected_amps = jnp.vstack([apply_low_pass_filter(row) for row in amps])

    return projected_amps

def generate_mem_comb_signal(omega, epsilon, chi, num_comb, num_points, phi, T, osc_drive_type):
    """
    Generate a signal composed of multiple frequency components.

    Parameters:
        omega (float): Scaling factor for the qubit drive (first row)
        epsilon (float): Scaling factor for the oscillator drive (second row)
        chi (float): dispersive coupling, frequency separation factor in the generated pulse
        num_comb (int): Number of frequency components in the frequency comb for qubit drive
        num_points (int): Number of time points in the output array.
        phi (jax.numpy.ndarray): Phase offsets for each frequency component of qubit drive(length = num_comb).

    Returns:
        jax.numpy.ndarray: A 2D array with two rows and `num_points` columns.
                           First row is qubit drive pulse
                           Second row is oscillator drive pulse
    """
    if osc_drive_type == 'linear':
        osc_pow = 1
    elif osc_drive_type == 'squeeze':
        osc_pow = 2

    # Ensure phi has the correct number of elements
    if len(phi) != num_comb:
        raise ValueError("The length of phi must be equal to num_comb.")

    # Time axis
    t = jnp.linspace(0, T, num_points)

    # Generate the qubit drive pulse (first row): omega * sum_n=0^num_comb (exp(-1j * n * chi * t + phi_n))
    qb_drive = jnp.zeros(num_points, dtype=jnp.complex64)
    for n in range(num_comb):
        qb_drive += jnp.exp(-1j * (n * chi * t + phi[n]))
    qb_drive *= omega

    # Generate the oscillator drive pulse (second row): epsilon * sum_n=0^1 (exp(-1j * n * chi * t))
    osc_drive = jnp.zeros(num_points, dtype=jnp.complex64)
    for n in range(2):
        osc_drive += jnp.exp(-1j * n * osc_pow* chi * t)
    osc_drive *= epsilon

    # Combine the rows into a 2D array
    result = jnp.vstack((osc_drive, qb_drive))

    return result

def generate_qubit_pulse(omega1, omega2, alpha1, alpha2, phi, levels1, levels2, num_points, T):

    if len(phi) != (levels1 + levels2):
        raise ValueError("The length of phi must be equal to levels1 + levels2.")

    t = jnp.linspace(0,T, num_points)

    qb_drive1 = jnp.zeros(num_points, dtype=jnp.complex64)
    for n in range(levels1):
        qb_drive1 += jnp.exp(-1j * (n* alpha1 * t + phi[n]))
    qb_drive1 *=omega1

    qb_drive2 = jnp.zeros(num_points, dtype=jnp.complex64)
    for n in range(levels2):
        qb_drive2 += jnp.exp(-1j * (n*alpha2 * t + phi[levels2+n]))
    qb_drive2 *=omega2

    amplitudes = jnp.vstack((qb_drive1, qb_drive2))

    return amplitudes

def build_ham(amps, params):

    a, adag = dq.destroy(params['ncav']), dq.create(params['ncav'])
    t, tdag = dq.destroy(params['ntr']), dq.create(params['ntr'])
    idcav = dq.eye(params['ncav'])
    idtr = dq.eye(params['ntr'])
    
    if params['osc_drive'] == 'linear':
        osc_pow = 1
    elif params['osc_drive']=='squeeze':
        osc_pow = 2

    # time-dependent Hamiltonian
    # (sum of  piece-wise constant Hamiltonians and of the static Hamiltonian)
    Hcr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(dq.powm(a,osc_pow) + dq.powm(adag,osc_pow), idtr))
    Hci = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor((dq.powm(a,osc_pow) - dq.powm(adag,osc_pow)), idtr))
    Htr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(idcav, t+tdag))
    Hti = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor(idcav, (t - tdag)))
    H = params['H0'] + Hcr + Hci + Htr + Hti

    return H

def evolution(H, psi0, tsave, exp_ops):
    """
    Solve the time-dependent Schrödinger equation for a given Hamiltonian and initial state,
    also calculates the given expectation operator with a list of times where to save

    Parameters:
        H (dq.QArray): Time-dependent Hamiltonian of the system.
        psi0 (dq.QArray): Initial state vector of the system.
        tsave (jnp.ndarray): Array of time points at which to save the results.
        exp_ops (list of dq.QArray): List of operators for which to compute expectation values.

    Returns:
        dq.SESolveResult: Result object containing the time evolution of the system and expectation values.
    """
        
    options = dq.Options(progress_meter = None)
    solver = dq.method.Tsit5(max_steps = int(1e9))

    return dq.sesolve(H, psi0, tsave, exp_ops = exp_ops, options = options, method = solver)


def get_evolution_result(amps, params):

    H = build_ham(amps, params)
    evo_result = evolution(H, params['psi0'], params['tsave'], params['exp_ops'])

    return evo_result

@jit
def compute_fidelity_loss(evo_result):

    """
    Computes average gate fidelity for the given amps and params

    Parameters:
        amps (jnp.ndarray): Input array of amplitudes (can have complex values).
        params (dict): Dictionary containing various parameters required for the simulation

    Returns:
        float: Average gate fidelity loss
    """

    return 1 - sum(evo_result.expects[i, i, -1].real for i in range(evo_result.expects.shape[0])) / evo_result.expects.shape[0]

@jit
def compute_smoothness_loss(amps):
    """
    Compute the smoothness loss for the given amplitudes.
    
    Parameters:
        amps (jnp.ndarray): Input array of amplitudes (can have complex values).
        weight (float): Weight factor for the smoothness loss (default: 1e-4).

    Returns:
        float: Smoothness loss value.

    ## Can include second derivative also by including jnp.diff(smoothness)
    """
    smoothness = jnp.sum(jnp.abs(jnp.diff(jnp.pad(amps, (1, 1))))**2)
    smoothness /= (len(amps)+1)*(jnp.abs(jnp.max(amps) - jnp.min(amps))**2)
    return smoothness
    
def compute_evolution(amps, params):

    """
    Computes expectation operators that reflect the evolution of states in oscillator
    
    Parameters:
        amps (jnp.ndarray): Input array of amplitudes (can have complex values).
        params (dict): Dictionary containing various parameters required for the simulation

    Returns:
        jnp.ndarray: Array of expectation values reflecting the evolution of states in the oscillator.
    """
    # next version should have the ladder operators also pass as arguments    
    a, adag = dq.destroy(params['ncav']), dq.create(params['ncav'])
    t, tdag = dq.destroy(params['ntr']), dq.create(params['ntr'])
    idcav = dq.eye(params['ncav'])
    idtr = dq.eye(params['ntr'])

    if params['osc_drive'] == 'linear':
        osc_pow = 1
    elif params['osc_drive']=='squeeze':
        osc_pow = 2

    # time-dependent Hamiltonian
    # (sum of  piece-wise constant Hamiltonians and of the static Hamiltonian)
    Hcr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(dq.powm(a,osc_pow) + dq.powm(adag,osc_pow), idtr))
    Hci = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor((dq.powm(a,osc_pow) - dq.powm(adag,osc_pow)), idtr))
    Htr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(idcav, t+tdag))
    Hti = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor(idcav, (t - tdag)))
    H = params['H0'] + Hcr + Hci + Htr + Hti
    result = evolution(H, params['psi0'], params['tsave'], params['e_ops_evo'])

    return (result.expects).real

def compute_qb_evolution(amps, params):

    """
    Computes expectation operators that reflect the evolution of states in qubit
    
    Parameters:
        amps (jnp.ndarray): Input array of amplitudes (can have complex values).
        params (dict): Dictionary containing various parameters required for the simulation

    Returns:
        jnp.ndarray: Array of expectation values reflecting the evolution of states in the qubit
    """

    # next version should have the ladder operators also pass as arguments    
    a, adag = dq.destroy(params['ncav']), dq.create(params['ncav'])
    t, tdag = dq.destroy(params['ntr']), dq.create(params['ntr'])
    idcav = dq.eye(params['ncav'])
    idtr = dq.eye(params['ntr'])

    if params['osc_drive'] == 'linear':
        osc_pow = 1
    elif params['osc_drive']=='squeeze':
        osc_pow = 2

    # time-dependent Hamiltonian
    # (sum of  piece-wise constant Hamiltonians and of the static Hamiltonian)
    Hcr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(dq.powm(a,osc_pow) + dq.powm(adag,osc_pow), idtr))
    Hci = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor((dq.powm(a,osc_pow) - dq.powm(adag,osc_pow)), idtr))
    Htr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(idcav, t+tdag))
    Hti = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor(idcav, (t - tdag)))
    H = params['H0'] + Hcr + Hci + Htr + Hti
    result = evolution(H, params['psi0'], params['tsave'], params['e_ops_qb'])

    return (result.expects).real

def compute_states(amps, params):
    """
    Compute the time evolution of the states given the amplitudes and parameters

    Parameters:
        amps (jnp.ndarray): Input array of amplitudes (can have complex values).
        params (dict): Dictionary containing various parameters required for the simulation

    Returns:
        jnp.ndarray: Array of states reflecting the evolution of the system.
    """
    # next version should have the ladder operators also pass as arguments    
    a, adag = dq.destroy(params['ncav']), dq.create(params['ncav'])
    t, tdag = dq.destroy(params['ntr']), dq.create(params['ntr'])
    idcav = dq.eye(params['ncav'])
    idtr = dq.eye(params['ntr'])

    if params['osc_drive'] == 'linear':
        osc_pow = 1
    elif params['osc_drive']=='squeeze':
        osc_pow = 2

    # time-dependent Hamiltonian
    # (sum of  piece-wise constant Hamiltonians and of the static Hamiltonian)
    Hcr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(dq.powm(a,osc_pow) + dq.powm(adag,osc_pow), idtr))
    Hci = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor((dq.powm(a,osc_pow) - dq.powm(adag,osc_pow)), idtr))

    Htr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(idcav, t+tdag))
    Hti = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor(idcav, (t - tdag)))
    H = params['H0'] + Hcr + Hci + Htr + Hti

    result = evolution(H, params['psi0'], params['tsave'], params['e_ops_evo'])

    return result.states

def sqrtm_jax(A, tol=1e-5):
    """
    Compute the matrix square root using JAX.

    Parameters:
    A (jnp.ndarray or DenseQArray): Input matrix.
    tol (float): Tolerance below which eigenvalues are considered zero.

    Returns:
    jnp.ndarray: Matrix square root of the input matrix.
    """
    if isinstance(A, dq.qarrays.dense_qarray.DenseQArray):
        A = jnp.array(A.to_jax())  # Convert DenseQArray to JAX array

    # Ensure input matrix is valid
    if jnp.any(jnp.isnan(A)) or jnp.any(jnp.isinf(A)):
        raise ValueError("Input matrix contains NaNs or infinities")

    # Ensure input matrix is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix")

    # Ensure input matrix is Hermitian
    if not jnp.allclose(A, A.conj().T):
        raise ValueError("Input matrix must be Hermitian")

    # Compute the matrix square root
    eigvals, eigvecs = jnp.linalg.eigh(A)

    # Set small negative eigenvalues to zero
    eigvals = jnp.where(eigvals < tol, 0, eigvals)

    sqrt_eigvals = jnp.sqrt(eigvals)
    sqrt_A = eigvecs @ jnp.diag(sqrt_eigvals) @ jnp.conj(eigvecs.T)

    return sqrt_A

def _fidelity(rho, sigma):
    """Compute the fidelity between two density matrices."""
    
    sqrt_rho = sqrtm_jax(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = sqrtm_jax(product)

    return jnp.real(jnp.trace(sqrt_product)) ** 2

def _bures_angle_distance(rho, sigma):
    """Compute the Bures angle distance between two density matrices."""
    fid = _fidelity(rho, sigma)
    fid = jnp.clip(jnp.real(fid), 0.0, 1.0)  # Ensure fid is within [0, 1] to avoid numerical issues
    return jnp.arccos(jnp.sqrt(fid))

def bures_angle(amps, params):

    """
    Evaluate the distance metric of the pulse to the target state by computing the bures angle
    between the states during evolution at each time step

    Parameters:
    amps (ndarray): Optimized amplitude of shape (2, ntpulse-1) to be used to simulate the system.
    params (dict): The usual params dictionary with all necessary parameters for simulation.

    Returns:
    float: Average distance metric (bures angle) for the operation
    """

    result = compute_states(amps, params)
    
    bures_sum = 0
    num_states = len(params['tsave'])

    for idx in range(6):
        den_mat = dq.ptrace(result[idx], dims = (params['ncav'],params['ntr']), keep = (0,))
        for i in range(num_states - 1):
            bures_sum += _bures_angle_distance(den_mat[i].to_jax(), den_mat[i + 1].to_jax())
            
    bures_sum /= (6*(num_states-1))

    # We can also normalize bures angle to be between 0 and 1 by dividing it by pi/2*(num_states - 1)
    # This will be useful in including it in the optimization loss function
    
    return bures_sum

@jit
def _optimal_time_fid(psi1, psi2):
    return (jnp.abs(dq.dag(psi1) @ psi2)**2)[0,0]

def optimal_time_fidelity_loss(evo_result, target_states):

    """
    Evaluate loss function for the fidelity for optimal time pulse generation as well as 
    average gate fidelity for the specific operation with given weights for the two metrics
    
    Parameters:
    amps (ndarray): Optimized amplitude of shape (2, ntpulse-1) to be used to simulate the system.
    weights(ndarray): 2 element array with weights for the two metrics
    params (dict): The usual params dictionary with all necessary parameters for simulation.

    Returns:
    float: total loss function for the operation
    """

    states = evo_result.states
    tsteps = len(states[0,:])-1
    target_states = tuple(target_states)

    # fid=0
    # for idx in range(6):
    #     for i in range(tsteps):
    #         fid += _optimal_time_fid(target_states[idx], states[idx,i])
    # fid /= (6*tsteps)

    def accumulate_time(carry, i):
        """
        Inner loop over time steps (0..tsteps-1).

        carry is (partial_fid, idx, states, target_states).
        partial_fid   : partial fidelity sum
        idx           : which row we are currently processing [0..5]
        states        : array of shape (6, tsteps+1, ...)
        target_states : tuple of length 6
        i    : current time index
        """
        partial_fid, idx, states, target_states = carry

        # Instead of direct indexing target_states[idx], use lax.switch.
        # The function _optimal_time_fid returns a float.
        overlap = jax.lax.switch(
            idx,
            tuple(  # Each element is a tiny lambda so JAX can stage them out.
                lambda st=st: _optimal_time_fid(st, states[idx, i]) 
                for st in target_states
            )
        )
        partial_fid += overlap

        new_carry = (partial_fid, idx, states, target_states)
        return new_carry, None

    def accumulate_idx(carry, idx):
        """
        Outer loop over idx in [0..5].

        carry is (partial_fid, states, target_states).
        partial_fid   : partial fidelity sum
        states        : array of shape (6, tsteps+1, ...)
        target_states : tuple of length 6
        idx  : current row index
        """
        partial_fid, states, target_states = carry

        # Number of valid time steps
        tsteps = states.shape[1] - 1

        # Inner loop's initial carry (zero out row's partial fidelity)
        init_for_time = (0.0, idx, states, target_states)

        # Scan over time steps 0..(tsteps-1)
        final_time_carry, _ = jax.lax.scan(
            accumulate_time,
            init_for_time,
            jnp.arange(tsteps)
        )

        row_fid, _, _, _ = final_time_carry
        partial_fid += row_fid

        new_carry = (partial_fid, states, target_states)
        return new_carry, None

    # Outer loop over idx in [0..5]
    init_for_idx = (0.0, states, target_states)
    final_idx_carry, _ = jax.lax.scan(
        accumulate_idx,
        init_for_idx,
        jnp.arange(6)
    )
    total_fid, _, _ = final_idx_carry

    fid = total_fid/(6*tsteps)

    return (1 - fid)

# The following function is incomplete, currently it returns all indices where pops<0.01
def find_maxoccupied_statenum(pops):
    """
    Find the maximum occupied state number for each time step in the simulation.

    Parameters:
    pops (ndarray): Population of each state for all psi0 at each time step in the simulation.

    Returns:
    ndarray: Array of tuples (psi0num, lowest even numbered state number which has pops<0.01, timestep number)
    """
    idx = float('inf')

    for i in np.arange(pops.shape[1], 0, -1):
        if np.max(pops[:,i,:])<0.01:
            idx = i

    return idx

# This function does not work - there is some problem with differentiation
# @functools.partial(jit, static_argnums=[1,2])
def optimal_distance_loss(amps, weights, params):

    """
    Evaluate loss function for the fidelity for optimal time pulse generation as well as 
    average gate fidelity for the specific operation with given weights for the two metrics
    
    Parameters:
    amps (ndarray): Optimized amplitude of shape (2, ntpulse-1) to be used to simulate the system.
    weights(ndarray): 2 element array with weights for the two metrics
    params (dict): The usual params dictionary with all necessary parameters for simulation.

    Returns:
    float: total loss function for the operation
    """
    # Convert the tuple back to a dictionary
    # params = dict(params_tuple)

    # This function includes both average gate fidelity and optimal time loss      
    a, adag = dq.destroy(params['ncav']), dq.create(params['ncav'])
    t, tdag = dq.destroy(params['ntr']), dq.create(params['ntr'])
    idcav = dq.eye(params['ncav'])
    idtr = dq.eye(params['ntr'])

    if params['osc_drive'] == 'linear':
        osc_pow = 1
    elif params['osc_drive']=='squeeze':
        osc_pow = 2

    # time-dependent Hamiltonian
    # (sum of  piece-wise constant Hamiltonians and of the static Hamiltonian)
    Hcr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(dq.powm(a,osc_pow) + dq.powm(adag,osc_pow), idtr))
    Hci = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[0,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor((dq.powm(a,osc_pow) - dq.powm(adag,osc_pow)), idtr))
    Htr = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.real(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), dq.tensor(idcav, t+tdag))
    Hti = dq.pwc(params['tpulse'], jnp.real(flattop_gaussian_pulse(jnp.imag(amps[1,:]), params['max_amp'], rise_fraction = 0.05)), -1j * dq.tensor(idcav, (t - tdag)))
    H = params['H0'] + Hcr + Hci + Htr + Hti

    result = evolution(H, params['psi0'], params['tsave'], params['exp_ops'])
    states = result.states

    target_states = params['target_states']

    fid=0
    for idx in range(6):
        den_mat = dq.ptrace(states[idx], dims = (params['ncav'],params['ntr']), keep = (0,))
        target_den = dq.ptrace(target_states[idx], dims = (params['ncav'],params['ntr']), keep = (0,))
        for i in range(len(params['tsave'])-1):
            # huh = _bures_angle_distance(den_mat[i].to_jax(), target_den.to_jax())
            fid += _bures_angle_distance(den_mat[i].to_jax(), target_den.to_jax())
    fid /= (6*(len(params['tsave'])-1))
    
    # Vectorized computation of fidelity - which possibly works, need to verify
    # fid = jnp.mean(jnp.abs(dq.dag(target_states) @ states[:, :-1])**2, axis=(1, 2))

    avg_gate_fidelity = sum(result.expects[i, i, -1].real for i in range(6)) / 6    

    return jnp.real((weights[0] * (1 - avg_gate_fidelity) + weights[1] * (fid))/(weights[0] + weights[1]))

@jit
def _fubini_study_distance_metric(psi1, psi2):
    """ calculates the Fubini-Study distance metric between two pure states"""
    return 2*jnp.arccos(jnp.abs(psi1.dag() @ psi2))[0,0]

def fubini_study_velocity(states):
    """ Calculates the total Fubini-Study distance metric for a states evolution
    corresponding to a given set of amplitudes and parameters"""

    num_states = states.shape[1] - 2

    velocity=0
    for idx in range(6):
        for i in range(num_states):
            velocity += _fubini_study_distance_metric(states[idx,i], states[idx,i+1])

    return velocity/(6*num_states)

def fubini_study_velocity_for_plot(states):

    num_states = states.shape[1] - 2
    
    velocity = np.empty(num_states)
    for i in range(num_states):
        vel = 0
        for idx in range(6):
            vel += _fubini_study_distance_metric(states[idx,i], states[idx,i+1])
        velocity[i] = vel/6
    
    return velocity

def fubini_study_velocity_variance(states):
    """
    Calculates the variance of the Fubini-Study distance metric for a states evolution
    corresponding to a given set of amplitudes and parameters.

    The distance is defined as the Fubini-Study distance between consecutive states,
    computed by _fubini_study_distance_metric. The variance is calculated as:
    
        variance = E[d^2] - (E[d])^2

    where E[d] is the mean distance over all measurements.
    """

    num_states = states.shape[1] - 2
    # Total number of distance measurements (using the same index range as the original)
    N = 6 * (num_states)
    
    sum_d = 0.0
    sum_d_sq = 0.0
    
    for idx in range(6):
        for i in range(num_states):
            d = _fubini_study_distance_metric(states[idx, i], states[idx, i+1])
            sum_d += d
            sum_d_sq += d**2
            
    mean_d = sum_d / N
    variance = (sum_d_sq / N) - mean_d**2
    
    return variance

def fubini_study_acceleration(states):
    """ Calculates the total Fubini-Study distance metric for a states evolution
    corresponding to a given set of amplitudes and parameters"""
    num_states = states.shape[1] - 2

    acceleration = 0
    for idx in range(6):
        for i in range(num_states):
            acceleration += jnp.abs(_fubini_study_distance_metric(states[idx,i], states[idx,i+1]) - _fubini_study_distance_metric(states[idx,i+1], states[idx,i+2]))

    return acceleration/(6*(num_states))

def fubini_study_acceleration_variance(states):
    """ Calculates the variance of the Fubini-Study velocity for a states evolution
    corresponding to a given set of amplitudes and parameters.
    
    The velocity is defined as the absolute difference between the Fubini-Study distance
    metrics computed on consecutive state pairs. The variance is calculated as:
        variance = (E[v^2] - (E[v])^2)
    where E[v] is the mean velocity.
    """

    num_states = states.shape[1] - 2
    # Number of velocity measurements: 6 rows and (len(params['tsave']) - 2) per row.
    N = 6 * num_states
    
    sum_velocity = 0.0
    sum_velocity_sq = 0.0
    
    for idx in range(6):
        for i in range(num_states):
            # Compute the velocity for this step.
            v = jnp.abs(
                _fubini_study_distance_metric(states[idx, i], states[idx, i+1]) -
                _fubini_study_distance_metric(states[idx, i+1], states[idx, i+2])
            )
            sum_velocity += v
            sum_velocity_sq += v**2

    mean_velocity = sum_velocity / N
    variance = (sum_velocity_sq / N) - mean_velocity**2

    return variance

@jit
def optimal_pure_velocity_loss(evo_result):
    """Calculate a loss function that can be used to minimize the distance travelled during 
    the quantum evolution for a given set of amplitudes and parameters, weights indicate
    weightage to be given to avg gate fidelity and total distance losses"""

    states = evo_result.states
    tsteps = len(states[0,:]) - 1

    def accumulate_over_time(carry, i):
        """
        Inner scan function over time steps.
        carry is (row_dist_sum, idx, states).
        i is the current time index.
        """
        row_dist_sum, idx, states = carry

        # Add the distance metric for states[idx, i] and states[idx, i+1]
        row_dist_sum += _fubini_study_distance_metric(states[idx, i], states[idx, i + 1])

        new_carry = (row_dist_sum, idx, states)
        return new_carry, None  # No per-step output

    def accumulate_over_idx(carry, idx):
        """
        Outer scan function over the row index (0 to 5).
        carry is (total_dist, states).
        idx is the current row index.
        """
        total_dist, states = carry

        # For each row, initialize the inner loop's carry
        # row_dist_sum = 0.0, store idx, pass in states
        init_for_time = (0.0, idx, states)

        # Inner scan over the time dimension
        final_row_carry, _ = jax.lax.scan(
            accumulate_over_time,
            init_for_time,
            jnp.arange(states.shape[1] - 1)  # shape[1] == tsteps+1, so range is (tsteps)
        )

        # final_row_carry is (row_dist_sum, idx, states)
        row_dist_sum, _, _ = final_row_carry

        # Accumulate row_dist_sum into total_dist
        new_carry = (total_dist + row_dist_sum, states)
        return new_carry, None
    
    init_for_idx = (0, states)
    final_carry, _ = jax.lax.scan(accumulate_over_idx, init_for_idx, jnp.arange(6))

    # final_carry = (final_total, states)
    final_total, _ = final_carry

    # Divide by (6 * tsteps) as in your original loop
    dist = final_total / (6.0 * tsteps)
        
    return dist

@jit
def optimal_pure_velocity_variance(evo_result):
    """
    Calculate the variance of the Fubini-Study distance metric for a states evolution
    corresponding to a given set of amplitudes and parameters.
    
    The variance is computed as:
        variance = (E[d^2] - (E[d])^2)
    where d is the Fubini-Study distance between consecutive states.
    """

    states = evo_result.states
    tsteps = len(states[0, :]) - 1  # Number of time steps for which distances are computed

    def accumulate_over_time(carry, i):
        """
        Inner scan function over time steps.
        carry is (row_sum, row_sum_sq, idx, states),
        where row_sum accumulates the distances and row_sum_sq accumulates their squares.
        i is the current time index.
        """
        row_sum, row_sum_sq, idx, states = carry

        # Compute the distance for states[idx, i] and states[idx, i+1]
        d = _fubini_study_distance_metric(states[idx, i], states[idx, i + 1])
        
        # Update the accumulators
        row_sum += d
        row_sum_sq += d ** 2

        new_carry = (row_sum, row_sum_sq, idx, states)
        return new_carry, None  # No per-step output

    def accumulate_over_idx(carry, idx):
        """
        Outer scan function over the row index (0 to 5).
        carry is (total_sum, total_sum_sq, states).
        idx is the current row index.
        """
        total_sum, total_sum_sq, states = carry

        # For each row, initialize the inner loop's carry:
        # row_sum = 0.0, row_sum_sq = 0.0, along with idx and states.
        init_for_time = (0.0, 0.0, idx, states)

        # Inner scan over the time dimension.
        final_row_carry, _ = jax.lax.scan(
            accumulate_over_time,
            init_for_time,
            jnp.arange(states.shape[1] - 1)  # (tsteps) iterations
        )

        row_sum, row_sum_sq, _, _ = final_row_carry

        # Accumulate the row's totals into the global totals.
        new_total_sum = total_sum + row_sum
        new_total_sum_sq = total_sum_sq + row_sum_sq

        new_carry = (new_total_sum, new_total_sum_sq, states)
        return new_carry, None

    # Initialize the outer scan with total_sum and total_sum_sq set to 0.0
    init_for_idx = (0.0, 0.0, states)
    final_carry, _ = jax.lax.scan(accumulate_over_idx, init_for_idx, jnp.arange(6))
    total_sum, total_sum_sq, _ = final_carry

    # Total number of distance measurements is 6 * tsteps
    N = 6.0 * tsteps

    # Compute the mean distance and then the variance.
    mean_dist = total_sum / N
    variance = (total_sum_sq / N) - mean_dist ** 2

    return variance

@jit
def _forbidden_cost(psi1, psi2):
    return (jnp.abs(psi1.dag() @ psi2)**2)[0,0]

# @functools.partial(jit, static_argnums=[1,2])
def forbidden_state_loss(evo_result, weights, forbidden_states):
    """
    Calculates a loss function that can be used to minimize population in given forbidden states in params
    
    Parameters:
        amps (jnp.ndarray): Input array of amplitudes (can have complex values).
        weights (list): List of weights for the average gate fidelity loss and forbidden state population loss.
                        weights[0] is the weight for the average gate fidelity loss.
                        weights[1] is a list of weights for each forbidden state.
        params (dict): Dictionary containing various parameters required for the simulation
    
    Returns:
        float: Total loss value combining the average gate fidelity loss and forbidden state population loss                    
    """

    states = evo_result.states
    forbidden_states = tuple(forbidden_states)
    weights = jnp.array(weights)
    tsteps = len(states[0,:])-2

    # loss = 0
    # for forbid_idx, forbidden_state in enumerate(forbidden_states):
    #     for idx in range(6):
    #         for i in range(tsteps):
    #             loss += weights[forbid_idx]*jnp.abs(forbidden_state.dag() @ states[idx,i])**2

    # loss /= (6*len(forbidden_states)*(tsteps))

    def accumulate_time(carry, i):
        """
        Inner scan function over time steps (0..tsteps-1).

        carry is:
        (partial_loss, states, forbidden_states, weights, forbid_idx, idx)

        i is the current time-step index.
        """
        partial_loss, states, forbidden_states, weights, forbid_idx, idx = carry

        # Compute the contribution for this (forbid_idx, idx, i)
        # .dag() is called on the forbidden state, then we do a matrix/vector product with states[idx, i].
        # contribution = weights[forbid_idx] * _forbidden_cost(forbidden_states[forbid_idx], states[idx,i])
        contribution = jnp.take(weights, forbid_idx) * _forbidden_cost(
        jax.lax.switch(forbid_idx, tuple(lambda fs=fs: fs for fs in forbidden_states)),
        states[idx, i])

        partial_loss = partial_loss + contribution
        new_carry = (partial_loss, states, forbidden_states, weights, forbid_idx, idx)
        return new_carry, None  # We don't need a per-step output
    
    def accumulate_idx(carry, idx):
        """
        Middle scan function over idx (0..5).

        carry is:
        (partial_loss, states, forbidden_states, weights, forbid_idx)
        """
        partial_loss, states, forbidden_states, weights, forbid_idx = carry

        # We'll run a scan over time steps i in [0..tsteps-1].
        # tsteps = states.shape[1] - 2, matching your original logic.
        tsteps = states.shape[1] - 2

        init_for_time = (0.0, states, forbidden_states, weights, forbid_idx, idx)
        final_time_carry, _ = jax.lax.scan(
            accumulate_time, init_for_time, jnp.arange(tsteps)
        )

        row_loss, _, _, _, _, _ = final_time_carry

        # Add this row's total to our partial_loss so far
        partial_loss = partial_loss + row_loss
        new_carry = (partial_loss, states, forbidden_states, weights, forbid_idx)
        return new_carry, None


    def accumulate_forbid(carry, forbid_idx):
        """
        Outer scan function over forbid_idx (i.e. enumerating forbidden_states).

        carry is:
        (partial_loss, states, forbidden_states, weights)
        """
        partial_loss, states, forbidden_states, weights = carry

        # Now we scan over the 6 rows: idx in [0..5]
        init_for_idx = (0.0, states, forbidden_states, weights, forbid_idx)
        final_idx_carry, _ = jax.lax.scan(accumulate_idx, init_for_idx, jnp.arange(6))

        idx_loss, _, _, _, _ = final_idx_carry
        partial_loss = partial_loss + idx_loss

        new_carry = (partial_loss, states, forbidden_states, weights)
        return new_carry, None

    # Initialize the total loss to 0.0
    init_carry = (0.0, states, forbidden_states, weights)
    
    # Outer scan over each forbidden_state index
    final_carry, _ = jax.lax.scan(
        accumulate_forbid,
        init_carry,
        jnp.arange(len(forbidden_states))
    )
    total_loss, _, _, _ = final_carry

    loss = total_loss/(6*len(forbidden_states)*(tsteps))
    
    return loss


def pulse_modulation(params):
    """
    Apply the flattop Gaussian pulse modulation to the input amplitudes.

    Parameters:
        mod_amps (jnp.ndarray): Input array of amplitudes with two rows and N columns.
        amplitude (float): Maximum amplitude of the modulated pulse.
        rise_fraction (float): Fraction of the total pulse length used for the Gaussian rise/fall (default: 0.05).

    Returns:
        jnp.ndarray: Array of modulated amplitudes with the same shape as the input array.
    """
    mod_amps = project_amps(params['best_amps'], params['max_freq'])
    # Apply the flattop Gaussian pulse modulation to each row separately
    modulated_osc = flattop_gaussian_pulse(mod_amps[0, :], params['max_amp'])
    modulated_qb = flattop_gaussian_pulse(mod_amps[1, :], params['max_amp'])

    # Stack the modulated rows to form the output array
    modulated_amps = jnp.vstack((modulated_osc, modulated_qb))

    return modulated_amps

@jit
def optimal_pure_acceleration_loss(evo_result):
    """Calculate a loss function that can be used to minimize the change in velocity during 
    the quantum evolution for a given set of amplitudes and parameters, weights indicate
    weightage to be given to avg gate fidelity and total velocity losses"""

    states = evo_result.states
    tsteps = len(states[0,:]) - 3

    def accumulate_over_time(carry, i):
        """
        Inner scan function over time steps.
        carry is (row_dist_sum, idx, states).
        i is the current time index.
        """
        row_dist_sum, idx, states = carry
        dist1 = _fubini_study_distance_metric(states[idx, i], states[idx, i + 1])
        dist2 = _fubini_study_distance_metric(states[idx, i + 1], states[idx, i + 2])

        # Add the distance metric for states[idx, i] and states[idx, i+1]

        row_dist_sum += jnp.abs(dist1 - dist2)**2

        new_carry = (row_dist_sum, idx, states)
        return new_carry, None  # No per-step output

    def accumulate_over_idx(carry, idx):
        """
        Outer scan function over the row index (0 to 5).
        carry is (total_dist, states).
        idx is the current row index.
        """
        total_velocity, states = carry

        # For each row, initialize the inner loop's carry:
        # row_dist_sum = 0.0, store idx, pass in states
        init_for_time = (0.0, idx, states)

        # Inner scan over the time dimension
        final_row_carry, _ = jax.lax.scan(
            accumulate_over_time,
            init_for_time,
            jnp.arange(tsteps)  # shape[1] == tsteps+1, so range is (tsteps)
        )

        # final_row_carry is (row_dist_sum, idx, states)
        row_dist_sum, _, _ = final_row_carry

        # Accumulate row_dist_sum into total_dist
        new_carry = (total_velocity + row_dist_sum, states)
        return new_carry, None
    
    init_for_idx = (0, states)
    final_carry, _ = jax.lax.scan(accumulate_over_idx, init_for_idx, jnp.arange(6))

    # final_carry = (final_total, states)
    final_total, _ = final_carry

    # Divide by (6 * tsteps) as in your original loop
    velocity = final_total / (6.0 * tsteps)

    return velocity

@jit
def optimal_pure_acceleration_variance(evo_result):
    """
    Calculate the variance of the velocity metric during quantum evolution.
    Velocity is defined as the absolute difference of the Fubini study distance metrics:
        velocity = | _fubini_study_distance_metric(state[i], state[i+1])
                   - _fubini_study_distance_metric(state[i+1], state[i+2]) |
    The variance is computed as the average of the squared deviations from the mean velocity.
    """
    states = evo_result.states
    # tsteps is computed as len(states[0,:]) - 3 as in the original code.
    tsteps = len(states[0, :]) - 3

    def accumulate_over_time(carry, i):
        """
        Inner scan function over time steps.
        carry is a tuple (row_sum, row_sum_sq, idx, states)
        where row_sum accumulates the sum of velocities for the current row,
        row_sum_sq accumulates the sum of squares of velocities.
        """
        row_sum, row_sum_sq, idx, states = carry

        # Compute the two distance metrics
        dist1 = _fubini_study_distance_metric(states[idx, i], states[idx, i + 1])
        dist2 = _fubini_study_distance_metric(states[idx, i + 1], states[idx, i + 2])
        # Compute velocity as the absolute difference
        v = jnp.abs(dist1 - dist2)

        # Update the running sums
        row_sum += v
        row_sum_sq += v * v

        new_carry = (row_sum, row_sum_sq, idx, states)
        return new_carry, None  # No per-step output

    def accumulate_over_idx(carry, idx):
        """
        Outer scan function over the row index (0 to 5).
        carry is a tuple (total_sum, total_sum_sq, states)
        where total_sum and total_sum_sq accumulate the sums for all rows.
        """
        total_sum, total_sum_sq, states = carry

        # Initialize inner loop's carry for a specific row:
        # row_sum = 0.0, row_sum_sq = 0.0, and pass in idx and states
        init_for_time = (0.0, 0.0, idx, states)

        # Inner scan over time steps for this row.
        final_row_carry, _ = jax.lax.scan(
            accumulate_over_time,
            init_for_time,
            jnp.arange(tsteps)
        )
        row_sum, row_sum_sq, _, _ = final_row_carry

        # Accumulate the results for this row into the total sums.
        new_total_sum = total_sum + row_sum
        new_total_sum_sq = total_sum_sq + row_sum_sq

        new_carry = (new_total_sum, new_total_sum_sq, states)
        return new_carry, None

    # Initialize the outer loop. There are 6 rows (0 to 5).
    init_for_idx = (0.0, 0.0, states)
    final_carry, _ = jax.lax.scan(accumulate_over_idx, init_for_idx, jnp.arange(6))
    total_sum, total_sum_sq, _ = final_carry

    # Total number of velocity measurements is the product of number of rows and time steps.
    N = 6.0 * tsteps

    # Compute the mean velocity.
    mean_velocity = total_sum / N
    # Compute variance using E[v^2] - (E[v])^2.
    variance = (total_sum_sq / N) - mean_velocity ** 2

    return variance


def target_evolution_loss(evo_states, tgt_states, hilbert_dim):

    # num_states = len(evo_states[0,:])

    # for idx in range(6):
    #     for i in range(num_states):
    #         fid += jnp.abs(evo_states[idx,i].dag() @ tgt_states[idx*num_states + i])**2

    # fid /= 6*num_states

    # return 1 - fid
    """
    Compute loss = 1 - average fidelity between evo_states and tgt_states.

    Both evo_states and tgt_states have shape (6, n). For each element we compute
    the fidelity as |⟨evo|tgt⟩|^2. Then we average over all 6*n elements.
    """
    # Define a function to compute fidelity for one pair of states.
    def fidelity(evo, tgt):
        if hasattr(evo, "to_jax"):
            evo = evo.to_jax()
        if hasattr(tgt, "to_jax"):
            tgt = tgt.to_jax()
        return jnp.abs(jnp.vdot(evo, tgt))**2
    
    if isinstance(tgt_states, list):
        tgt_states = jnp.stack([ts.to_jax() if hasattr(ts, "to_jax") else ts for ts in tgt_states])
        tgt_states = tgt_states.reshape(6, evo_states.shape[1], hilbert_dim, 1)

    # Vectorize over the inner time axis for each row.
    fidelity_over_time = jax.vmap(fidelity, in_axes=(0, 0))
    # Now vectorize over the 6 initial states
    fidelities = jax.vmap(fidelity_over_time)(evo_states, tgt_states)

    # Compute the mean fidelity over all indices.
    mean_fid = jnp.mean(fidelities)

    return 1 - mean_fid

# Are these functions making use of jax jit? 
def qb_freq_robustness(amps, dets, params):

    """
    Evaluate the robustness of the pulse to qubit frequency detuning by computing the average gate fidelity
    for different detuning values.

    Parameters:
    amps (ndarray): Optimized amplitude of shape (2, ntpulse-1) to be used to simulate the system.
    dets (ndarray): Detuning values for the qubit frequency to be used to simulate the system.
                    Ideally, this should have both negative and positive values.
                    Each value of det will be used for a different simulation with an addition of
                    det[i] * dq.tensor(idcav, t @ tdag) to params['H0'].
    params (dict): The usual params dictionary with all necessary parameters for simulation.

    Returns:
    ndarray: Average gate fidelity for binomial gate operations for the different detuning values,
             same shape as dets.
    """

    fids = np.empty(len(dets))
    params_int = params.copy()

    # Define the operators
    idcav = dq.eye(params_int['ncav'])
    t, tdag = dq.destroy(params_int['ntr']), dq.create(params_int['ntr'])
    orig_H0 = params_int['H0']

    for i, det in enumerate(dets):
        # Update H0 with the detuning value
        H0_detuned = orig_H0 + det * dq.tensor(idcav, tdag @ t)
        params_int['H0'] = H0_detuned

        evo_result = get_evolution_result(amps, params_int)

        # Compute fidelity for the current detuning value
        fids[i] = 1- compute_fidelity_loss(evo_result)

    return fids

# Are you in any way taking advantage of jax jit
def chi_robustness(amps, dets, params):

    """
    Evaluate the robustness of the pulse to chi(dispersive coupling) detuning by computing the average gate fidelity
    for different detuning values.

    Parameters:
    amps (ndarray): Optimized amplitude of shape (2, ntpulse-1) to be used to simulate the system.
    dets (ndarray): Detuning values for the qubit frequency to be used to simulate the system.
                    Ideally, this should have both negative and positive values.
                    Each value of det will be used for a different simulation with an addition of
                    det[i] * dq.tensor(a @ adag, t @ tdag) to params['H0'].
    params (dict): The usual params dictionary with all necessary parameters for simulation.

    Returns:
    ndarray: Average gate fidelity for binomial gate operations for the different detuning values,
             same shape as dets.
    """

    fids = np.empty(len(dets))
    params_int = params.copy()

    # Define the operators
    a, adag = dq.destroy(params_int['ncav']), dq.create(params_int['ncav'])
    t, tdag = dq.destroy(params_int['ntr']), dq.create(params_int['ntr'])
    orig_H0 = params_int['H0']

    for i, det in enumerate(dets):
        # Update H0 with the detuning value
        H0_detuned = orig_H0 + det * dq.tensor(adag @ a, tdag @ t)
        params_int['H0'] = H0_detuned

        evo_result = get_evolution_result(amps, params_int)

        # Compute fidelity for the current detuning value
        fids[i] = 1 - compute_fidelity_loss(evo_result)

    return fids

def noise_robustness(amps, noise_percentages, params):
    """
    Evaluate the robustness of the pulse to noise by computing the average gate fidelity
    for different noise percentages.

    Parameters:
    amps (ndarray): Optimized amplitude of shape (2, ntpulse-1) to be used to simulate the system.
    noise_percentages (ndarray): Array of noise percentages to be added to the amplitudes.
                                 Each value represents the percentage of noise to be added.
    params (dict): The usual params dictionary with all necessary parameters for simulation.

    Returns:
    ndarray: Average gate fidelity for binomial gate operations for the different noise percentages,
             same shape as noise_percentages.
    """  
    fids = np.empty(len(noise_percentages))
    original_amps = amps # Save the original amplitudes
    
    noise_osc = jax.random.uniform(jax.random.PRNGKey(101), shape=(params['ntpulse']-1, ), minval= -3, maxval= 3) \
        + 1j*jax.random.uniform(jax.random.PRNGKey(101), shape=(params['ntpulse']-1, ), minval= -3, maxval= 3)
    noise_qb = jax.random.uniform(jax.random.PRNGKey(101), shape=(params['ntpulse']-1, ), minval= -3, maxval= 3) \
        + 1j*jax.random.uniform(jax.random.PRNGKey(101), shape=(params['ntpulse']-1, ), minval= -3, maxval= 3)
    
    for i, noise_percentage in enumerate(noise_percentages):
        
        noise = (noise_percentage / 100.0) * jnp.array([noise_osc, noise_qb])
        noisy_amps = original_amps + noise

        evo_result = get_evolution_result(noisy_amps, params)

        # Compute fidelity for the current noisy amplitudes
        fids[i] = 1 - compute_fidelity_loss(evo_result)

    return fids

def fourier_transform(amps, dt = 4e-9):
    """
    Given a 2D array 'amps' of shape (2, tsteps), where each row
    is a complex-valued waveform sampled in time at 4 ns intervals,
    compute and return:
      - freqs: The frequency axis for the FFT (in Hz)
      - amps_fft: The FFT of each row of 'amps' (same shape as 'amps')

    amps[0, :] => qubit drive waveform
    amps[1, :] => oscillator drive waveform
    """
    
    # Number of time samples
    tsteps = amps.shape[1]
    
    # Compute the frequency axis; np.fft.fftfreq returns an array of length 'tsteps'
    # with frequencies arranged from 0 up to fs/2 and then negative frequencies.
    freqs = np.fft.fftfreq(tsteps, d=dt)
    
    # Compute the FFT along axis=1 so we transform each row (waveform) separately
    amps_fft = np.fft.fft(amps, axis=1)
    
    return freqs, amps_fft

def inverse_fourier_transform(freqs, waveforms_fft):
    """
    Given:
      - freqs: 1D array of length N, as returned by np.fft.fftfreq(N, d=dt)
      - waveforms_fft: 2D array of shape (2, N) containing frequency-domain
        data for two waveforms (each row is one waveform in frequency space)

    Returns:
      - times: 1D array of length N giving the time axis (0 to (N-1)*dt)
      - waveforms_time: 2D array (2, N) with the inverse Fourier transform
        of waveforms_fft (each row is one waveform in time space).
    
    Requirements:
      - waveforms_fft and freqs must have consistent lengths (N).
      - freqs must be the standard (unshifted) fftfreq array so that
        ifft() recovers the time-domain data correctly.
    """
    # Number of samples
    N = waveforms_fft.shape[1]
    if len(freqs) != N:
        raise ValueError("freqs and waveforms_fft must have matching lengths.")

    # Recover dt from the spacing between consecutive frequency bins
    # For an unshifted frequency array from np.fft.fftfreq(N, dt),
    # freq spacing = 1/(N*dt). Hence:
    df = freqs[1] - freqs[0]  # This should be 1/(N * dt)
    dt = 1.0 / (N * df)

    # Create time axis: samples are separated by dt
    times = np.arange(N) * dt

    # Compute the inverse FFT along axis=1
    waveforms_time = np.fft.ifft(waveforms_fft, axis=1)

    return times, waveforms_time

# @jax.jit
# def qb_et_loss(evo_states, ncav, ntr):
#     """
#     Compute the error-transparency cost 
    
#     Args:
#       evo_states: array of shape (d1, d2), dtype=Qobj
#         Pure joint state vectors at each initial state and timestep.
#       ncav: int
#         Oscillator Hilbert space dimension ncav
#       ntr: int
#         Qubit Hilbert space dimension
    
#     Returns:
#       Scalar real loss = mean_{i,j} Tr[(rho_g - rho_e)^2].
#     """

#     proj_ground = dq.tensor(
#     dq.eye(ncav),
#     dq.proj(dq.basis(ntr, 0))
#     )
#     proj_excited = dq.tensor(
#         dq.eye(ncav),
#          dq.proj(dq.basis(ntr, 1))
#     )
    
#     d1, d2 = evo_states.shape[:2]

#     for i in range(d1):
#         for j in range(d2):
#             st_g = dq.ptrace((proj_ground @ evo_states[i,j,:,:] @ evo_states[i,j,:,:].dag())) 
#             st_e = dq.ptrace((proj_excited @ evo_states[i,j,:,:] @ evo_states[i,j,:,:].dag()))
#             fid += jnp.real(dq.trace((st_g - st_e) @ (st_g - st_e)))
    
#     fid /= (d1*d2)

#     return fid

def qb_et_loss(evo_states, ncav: int, ntr: int) -> jnp.ndarray:
    """
    JIT‑able, double‑loop via nested lax.scan over evo_states[i,j,:,:].

    evo_states: array of shape (d1, d2, dim, 1)  where evo_states[i,j,:,:] is
                the pure state vector psi_{i,j} (as a Qobj).
    ncav, ntr: static ints for building the projectors.
    """
    # build projectors once
    proj_g = dq.tensor(
        dq.eye(ncav),
        dq.proj(dq.basis(ntr, 0))
    )
    proj_e = dq.tensor(
        dq.eye(ncav),
        dq.proj(dq.basis(ntr, 1))
    )

    d1, d2 = evo_states.shape[:2]

    def _scan_over_j(fid_j, psi_ij):
        # form density matrix ρ = |ψ><ψ|
        ρ = psi_ij @ psi_ij.dag()
        # partial traces onto cavity space
        ρg = dq.ptrace(proj_g @ ρ , dims = (ncav,ntr), keep = (0,))
        ρe = dq.ptrace(proj_e @ ρ , dims = (ncav,ntr), keep = (0,))
        Δ  = ρg - ρe
        incr = jnp.real(dq.trace(Δ @ Δ))
        return fid_j + incr, None

    def _scan_over_i(fid_i, psi_row):
        # psi_row has shape (d2, dim, 1)
        fid_row, _ = jax.lax.scan(_scan_over_j, 0.0, psi_row)
        return fid_i + fid_row, None

    fid_total, _ = jax.lax.scan(_scan_over_i, 0.0, evo_states)
    return fid_total / (d1 * d2)

# incomplete function implementation
def generate_seeds(method, ntpulse, **kwargs):
    if method == 'mem_comb':
        omega = kwargs.get('omega')
        epsilon = kwargs.get('epsilon')
        chi = kwargs.get('chi')
        T = kwargs.get('T')
        num_comb = kwargs.get('num_comb')
        phi = kwargs.get('phi')
        num_points = kwargs.get('num_points')
        amps_seed = generate_comb_signal_squeezing(omega, epsilon, chi, num_comb, num_points, phi, T)
        amps = amps_seed

    elif method == 'seed_var':
        key = kwargs.get('key')
        seed_var = jax.random.uniform(jax.random.PRNGKey(key), shape=(ntpulse-1, ), minval=-1, maxval=1) + 1j*jax.random.uniform(jax.random.PRNGKey(key), shape=(ntpulse-1, ), minval=-1, maxval=1)
        omega = kwargs.get('omega')
        epsilon = kwargs.get('epsilon')
        chi = kwargs.get('chi')
        T = kwargs.get('T')
        num_comb = kwargs.get('num_comb')
        phi = kwargs.get('phi')
        num_points = kwargs.get('num_points')
        amps_seed = generate_comb_signal_squeezing(omega, epsilon, chi, num_comb, num_points, phi, T)
        amps_seed = kwargs.get('amps_seed')
        amps = amps_seed + jnp.array([seed_var, seed_var])

    elif method == 'random_pulse_shape':
        amps_qb_seed = jax.random.uniform(jax.random.PRNGKey(101), shape=(ntpulse-1, ), minval=-2, maxval=2) + 1j*jax.random.uniform(jax.random.PRNGKey(101), shape=(ntpulse-1, ), minval=-2, maxval=2)
        amps_osc_seed = jax.random.uniform(jax.random.PRNGKey(1101), shape=(ntpulse-1, ), minval=-3, maxval=3) + 1j*jax.random.uniform(jax.random.PRNGKey(1101), shape=(ntpulse-1, ), minval=-3, maxval=3)
        amps_seed = jnp.array([amps_qb_seed, amps_osc_seed])
        amps = jnp.array([amps_qb_seed, amps_osc_seed])

    elif method == 'load_from_file':
        directory = kwargs.get('directory')
        filename = kwargs.get('filename')
        full_path = os.path.join(directory, filename)
        loaded_file = np.load(full_path, allow_pickle=True)
        params = {key: loaded_file[key] for key in loaded_file.files}
        amps_seed = jnp.array(params['amps'])
        amps = amps_seed

    else:
        raise ValueError("Unknown method for generating seeds")

    return amps_seed, amps



# @functools.partial(jit, static_argnames=['weights', 'params'])
def grand_loss_calculator(amps, weights, params):

    loss = 0

    evo_result = get_evolution_result(amps, params)

    avg_gate_fidelity_loss = compute_fidelity_loss(evo_result)

    cnt=0
    loss += weights[cnt]*avg_gate_fidelity_loss
    cnt+=1

    if params['include_smoothness_loss']:
        loss += weights[cnt]*compute_smoothness_loss(amps)
        cnt+=1

    if params['include_pure_velocity_loss']:
        loss += weights[cnt]*optimal_pure_velocity_loss(evo_result)
        cnt+=1
    
    if params['include_pure_velocity_variance_loss']:
        loss += weights[cnt]*optimal_pure_velocity_variance(evo_result)
        cnt+=1

    if params['include_optimal_time_loss']:
        loss += weights[cnt]*optimal_time_fidelity_loss(evo_result, params['target_states'])
        cnt+=1

    if params['include_pure_acceleration_loss']:
        loss += weights[cnt]*optimal_pure_acceleration_loss(evo_result)
        cnt+=1

    if params['include_pure_acceleration_variance_loss']:
        loss += weights[cnt]*optimal_pure_acceleration_variance(evo_result)
        cnt+=1

    if params['include_target_evolution_dynamics']:
        loss += weights[cnt] * target_evolution_loss(evo_result.states, params['tgt_evo_states'], params['ncav']*params['ntr'])
        cnt+=1
    
    if params['include_qubit_et_loss']:
        loss += weights[cnt] * qb_et_loss(evo_result.states, params['ncav'], params['ntr'])
        cnt+=1

    if params['include_forbidden_states_loss']:
        loss+= weights[cnt]*forbidden_state_loss(evo_result, params['weights'][cnt+1], params['forbidden_states'])
        cnt+=1 

    # Sum only the numeric values, ignoring nested lists
    # total_sum = sum(w for w in weights if isinstance(w, (int, float)))

    # loss/= total_sum

    return loss, avg_gate_fidelity_loss


def loss_scaling(psi0, recovery, target_state, params):

    gammas = np.linspace(0, 0.01, 10)
    idtr = dq.eye(params['ntr'])
    a = dq.destroy(params['ncav'])
    H = build_ham(params['amps'], params)
    
    options = dq.Options(progress_meter = None)
    solver = dq.solver.Tsit5(max_steps = int(1e9))

    process_fidelities = np.zeros(len(gammas))

    for i, gamma in enumerate(gammas):
        c_ops = jnp.sqrt(gamma)*dq.tensor(a, idtr)
        evo_result = dq.mesolve(H, rho0 = psi0, tsave=params['tsave'], jump_ops = c_ops, Method = solver, options=options)

        fin_den = recovery @ evo_result.states[-1] @ recovery.dag()
        target_den = target_state @ target_state.dag()
        diff = fin_den - target_den
        process_fidelities[i] = jnp.real(dq.trace(diff @ diff))
        print(f"Gamma: {gamma}, Process Fidelity: {process_fidelities}")

    return gammas, process_fidelities

def et_improvement(psi0, target_state, proj, params):

    T1q = 60
    T2q = 90
    Tphiq = (1/T2q - 1/(2*T1q))**(-1)

    T1c = 200
    T2c = 300
    Tphic = (1/T2c - 1/(2*T1c))**(-1)

    a = dq.destroy(params['ncav'])
    adag = dq.create(params['ncav'])
    t = dq.destroy(params['ntr'])
    tdag = dq.create(params['ntr'])
    idcav = dq.eye(params['ncav'])
    idtr = dq.eye(params['ntr'])

    H = build_ham(params['amps'], params)

    c_ops = [jnp.sqrt(1/T1c)*dq.tensor(a, idtr), 
            # jnp.sqrt(1/Tphic)*dq.tensor(adag @ a, idtr),
            # jnp.sqrt(1/T1q)*dq.tensor(idcav, t),
            # jnp.sqrt(1/Tphiq)*dq.tensor(idcav, tdag @ t)
            ]
    
    options = dq.Options(progress_meter = None)
    solver = dq.method.Tsit5(max_steps = int(1e9))

    evo_result = dq.mesolve(H, rho0 = psi0, tsave = params['tsave'], jump_ops = c_ops, method = solver, options=options)

    fin_den = proj @ evo_result.states[-1]
    target_den = target_state @ target_state.dag()
    diff = fin_den - target_den
    process_fidelity = jnp.real(dq.trace(diff @ diff))
    print(f"Process Infidelity: {process_fidelity}")

