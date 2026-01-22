# optimizers.py
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- NEURAL NETWORK MODEL ---
class AsymmetricInitializer(tf.keras.initializers.Initializer):
    """
    Custom initializer that applies different scaling factors to the columns
    of a weight matrix. This allows initializing cavity and qubit drive
    outputs with different amplitude scales.
    """
    def __init__(self, scales, seed=None):
        """
        Args:
            scales (list or tuple): A list of scaling factors for each output column.
            seed (int, optional): Random seed for reproducibility.
        """
        self.scales = scales
        self.seed = seed
        self.base_initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)

    def __call__(self, shape, dtype=None):
        # `shape` is (fan_in, fan_out)
        initial_weights = self.base_initializer(shape, dtype=dtype)
        if shape[-1] != len(self.scales):
            raise ValueError(
                f"The number of scales ({len(self.scales)}) must match the "
                f"number of output units ({shape[-1]})."
            )
        scaling_vector = tf.constant(self.scales, dtype=dtype)
        return initial_weights * scaling_vector # Apply column-wise scaling

    def get_config(self):
        return {'scales': self.scales, 'seed': self.seed}

class PulseGenerator(tf.keras.Model):
    """
    A network that uses Fourier features to overcome spectral bias,
    with a small final hidden layer for SPSA tuning.
    """
    def __init__(self, n_pulse_points, max_amp, fourier_scale=10, output_scales=(1.0, 1.0, 1.0, 1.0)):
        super(PulseGenerator, self).__init__()
        self.max_amp = tf.constant(max_amp, dtype=tf.keras.backend.floatx())
        self.n_pulse_points = n_pulse_points

        # Fourier Feature mapping layer
        self.fourier_features = FourierFeatures(output_dim=128, scale=fourier_scale)
        
        # Main processing layers
        self.hidden_layer_1 = tf.keras.layers.Dense(128, activation='tanh')
        self.hidden_layer_2 = tf.keras.layers.Dense(64, activation='tanh')
        
        # Final hidden layer with 4 nodes for SPSA tuning
        self.final_hidden_layer = tf.keras.layers.Dense(4, activation='tanh', name='final_hidden_layer')
        
        # Custom initializer for the output layer
        output_initializer = AsymmetricInitializer(scales=output_scales)
        
        #  Output layer produces 4 components (I_c, Q_c, I_q, Q_q)
        self.output_layer = tf.keras.layers.Dense(
            4, 
            activation=None, 
            name='output_layer',
            kernel_initializer=output_initializer  # Use the custom initializer
        )

    def call(self, t):
        # Pass input through the full network architecture
        x = self.fourier_features(t)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.final_hidden_layer(x)
        envelopes = self.output_layer(x)
        
        # Unpack the I/Q components
        I_c, Q_c, I_q, Q_q = [envelopes[..., i] for i in range(4)]
        
        # Combine into complex-valued drives
        drive_c = tf.complex(I_c, Q_c)
        drive_q = tf.complex(I_q, Q_q)
        
        # Squeeze out the batch dimension
        amps = tf.stack([tf.squeeze(drive_c, axis=0), tf.squeeze(drive_q, axis=0)], axis=0)
        return amps

def fftfreq_tf(n, d=1.0):
    """TensorFlow implementation of numpy.fft.fftfreq."""
    n_int = tf.cast(n, tf.int32)
    k = tf.concat([tf.range((n_int + 1) // 2, dtype=tf.int32),
                   tf.range(-(n_int // 2), 0, dtype=tf.int32)], axis=0)
    k_float = tf.cast(k, tf.keras.backend.floatx())
    return k_float / (tf.cast(n, tf.keras.backend.floatx()) * tf.cast(d, tf.keras.backend.floatx()))

def low_pass_filter_tf(signal, max_freq, time_step):
    """Applies a low-pass filter using native TensorFlow operations."""
    signal_fft = tf.signal.fft(signal)
    # NOTE: Dtype is inferred from the signal
    freqs = fftfreq_tf(tf.shape(signal)[-1], d=time_step)
    
    mask = tf.abs(freqs) > max_freq
    signal_fft_filtered = tf.where(mask, tf.cast(0, signal_fft.dtype), signal_fft)
    return tf.signal.ifft(signal_fft_filtered)

def flattop_gaussian_pulse_tf(input_array, amplitude, rise_fraction=0.05):
    """Modulates a pulse with a flat-top Gaussian envelope using TensorFlow."""
    num_points = tf.shape(input_array)[-1]
    real_dtype = input_array.dtype.real_dtype
    
    # rise_points = tf.cast(tf.cast(num_points, real_dtype) * rise_fraction, tf.int32)
    rise_points = tf.constant(12, dtype=tf.int32)
    flat_points = num_points - 2 * rise_points

    sigma = tf.cast(rise_points, real_dtype) / 5.0

    start_val = tf.cast(0.0, real_dtype)
    end_val_rise = tf.cast(-5.0 * sigma, real_dtype)
    end_val_fall = tf.cast(5.0 * sigma, real_dtype)

    t_rise = tf.linspace(end_val_rise, start_val, rise_points)
    t_fall = tf.linspace(start_val, end_val_fall, rise_points)
    
    gaussian_rise = tf.exp(-tf.square(t_rise) / (2 * tf.square(sigma)))
    gaussian_fall = tf.exp(-tf.square(t_fall) / (2 * tf.square(sigma)))
    flat_top = tf.ones(flat_points, dtype=real_dtype)
    
    envelope = tf.concat([gaussian_rise, flat_top, gaussian_fall], axis=0)
    envelope_complex = tf.cast(envelope, dtype=input_array.dtype)
    
    real_part = tf.clip_by_value(tf.math.real(input_array), -amplitude, amplitude)
    imag_part = tf.clip_by_value(tf.math.imag(input_array), -amplitude, amplitude)

    clipped_input_array = tf.complex(real_part, imag_part)

    # print(f"DEBUG -> dtype of clipped_input_array: {clipped_input_array.dtype}")
    # print(f"DEBUG -> dtype of envelope_complex:   {envelope_complex.dtype}")
    
    return clipped_input_array * envelope_complex

class FourierFeatures(tf.keras.layers.Layer):
    """
    Maps a time input to a higher-dimensional vector of sinusoidal features
    to help the network learn high-frequency functions.
    """
    def __init__(self, output_dim, scale=10.0, **kwargs):
        super(FourierFeatures, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.scale = scale

    def build(self, input_shape):
        """Create the layer's weights, which will be managed by Keras."""
        self.b_matrix = self.add_weight(
            name='b_matrix',
            shape=(1, self.output_dim // 2),
            # Use an initializer to generate the random values just once
            initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),
            # This is crucial: make it part of the model's state but not trainable
            trainable=False
        )
        super().build(input_shape)

    def call(self, t):
        # Ensure constants use the correct precision
        pi = tf.constant(np.pi, dtype=tf.keras.backend.floatx())
        # Project the time input onto the random frequencies
        x_proj = (2 * pi * t) @ self.b_matrix
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

# --- MAIN OPTIMIZER CLASS ---
class HardwareAwareOptimizer:
    """
    Manages the end-to-end workflow for pulse optimization.
    """
    def __init__(self, params, fourier_scale=10, output_scales=(1.0, 1.0, 1.0, 1.0)):
        print("Initializing Optimizer...")
        self.params = params
        self.ntpulse = params.get('ntpulse', 251)
        self.max_amp = params.get('max_amp', 2*2*np.pi)
        
        # Pass the scaling factors to the PulseGenerator model
        self.model = PulseGenerator(
            self.ntpulse - 1, 
            self.max_amp, 
            fourier_scale=fourier_scale,
            output_scales=output_scales
        )
        print("Neural network model created.")

    def save_model_weights(self, filepath):
        """
        Saves the current model's weights to a file.
        
        Args:
            filepath (str): The path to save the weights file (e.g., 'pretrained.weights.h5').
        """
        self.model.save_weights(filepath)
        print(f"Model weights saved successfully to {filepath}")

    def load_model_weights(self, filepath):
        """
        Loads model weights from a file into the current model.
        The model architecture must be identical.
        
        Args:
            filepath (str): The path to the saved weights file.
        """
        # We must build the model first by calling it once
        time_array = self.params['tpulse'][:-1]
        dummy_input = tf.zeros((1, len(time_array), 1), dtype=tf.float32)
        self.model(dummy_input) # This call initializes the weights' shapes
        
        # Now we can load the saved weights
        self.model.load_weights(filepath)
        print(f"Model weights loaded successfully from {filepath}")

    def _apply_pulse_processing_tf(self, raw_amps):
        """
        Internal helper method to apply the full TF-native processing chain.
        """
        time_step = self.params['T'] / self.params['ntpulse']
        max_freq_hz = self.params.get('max_freq', 25) * 1e6 # assuming max_freq is in MHz

        drive_c_raw = raw_amps[0, :]
        drive_q_raw = raw_amps[1, :]

        # Apply low-pass filter
        filtered_c = low_pass_filter_tf(drive_c_raw, max_freq_hz, time_step)
        filtered_q = low_pass_filter_tf(drive_q_raw, max_freq_hz, time_step)

        # Apply flattop Gaussian envelope
        final_c = flattop_gaussian_pulse_tf(filtered_c, self.max_amp)
        final_q = flattop_gaussian_pulse_tf(filtered_q, self.max_amp)
        
        return tf.stack([final_c, final_q])
    
    def generate_pulse(self):
        """
        Generates the final, processed pulse array from the current model state.
        
        Returns:
            np.ndarray: The complex-valued, processed pulse array of shape (2, n_pulse_points).
        """

        print("Generating pulse from the trained model...")
        # Prepare the time vector input for the model
        time_array = self.params['tpulse'][:-1]
        time_input = tf.constant(time_array, dtype=tf.float32)
        time_input = tf.reshape(time_input, (1, len(time_array), 1))

        # Get the raw output from the neural network
        raw_amps = self.model(time_input)
        
        # Apply the same low-pass filter and envelope as in training
        processed_amps = self._apply_pulse_processing_tf(raw_amps)
        
        return processed_amps.numpy()

    def pre_train(self, ideal_pulse_arrays, time_array, epochs=1000, 
                  learning_rate=1e-3, print_every_x_steps=1000, 
                  target_mse = 1e-3, plot_live = True):
        """
        Pre-trains the NN to replicate a given ideal pulse.
        """
        print("\n--- Starting Pre-training ---")
        # The target pulse to imitate
        target_pulse = tf.constant(np.stack(ideal_pulse_arrays), dtype=tf.complex64)
        
        # Prepare the time vector for the model
        time_input = tf.constant(time_array, dtype=tf.float32)
        time_input = tf.reshape(time_input, (1, len(time_array), 1))

        # The user-provided learning_rate is now the initial value.
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=learning_rate,
        #     decay_steps=1000,  # How often to decay
        #     decay_rate=0.95)   # How much to decay by

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # if plot_live:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        line, = ax.plot([], [], '.-', label='Pre-training MSE Loss')
        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Live Pre-training Loss')
        ax.grid(True, which="both")
        ax.legend()

        losses = []

        for epoch in tqdm(range(epochs), desc="Pre-training"):
            with tf.GradientTape() as tape:
                raw_pulse = self.model(time_input)
                generated_pulse = self._apply_pulse_processing_tf(raw_pulse)
                #  Use Mean Squared Error for the loss
                loss = tf.reduce_mean(tf.abs(tf.square(generated_pulse - target_pulse)))
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            losses.append(loss.numpy())
            
            if epoch%1000 == 0:
                line.set_xdata(range(len(losses)))
                line.set_ydata(losses)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01) # Pause to allow the plot to update
                
            # Check if the user closed the plot window
            if not plt.fignum_exists(fig.number):
                print("Plot window closed. Stopping pre-training.")
                break

            # if epoch % print_every_x_steps == 0:
            #         print(f"\nEpoch {epoch}: Pre-training MSE Loss = {loss.numpy():.6f}")

            if loss <= target_mse:
                print(f"\nTarget MSE of {target_mse} reached at epoch {epoch}. Stopping pre-training.")
                break

        print(f"Pre-training finished. Final MSE Loss: {loss.numpy():.6f}")

    def fine_tune_simulation(self, loss_calculator_func, epochs=1000, 
                             learning_rate=1e-4, target_fidelity_loss=1e-3, 
                             update_plot_every=100):
        """
        Fine-tunes the NN in a simulated environment using backpropagation.
        """
        print("\n--- Starting Simulation Fine-tuning ---")
        time_array = self.params['tpulse'][:-1]
        time_input = tf.constant(time_array, dtype=tf.float32)
        time_input = tf.reshape(time_input, (1, len(time_array), 1))

        # Using a fixed learning rate as requested.
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Helper function for JAX calculation:
        def _get_loss_and_grad_from_jax_wrapper(amps_np):
            amps_jax = jnp.array(amps_np)
            
            # This correctly uses the functions and data from the Python scope
            loss, grad = jax.value_and_grad(loss_calculator_func)(amps_jax, self.params)
            
            return np.float32(loss), np.complex64(grad)

        # Define the custom gradient wrapper for TensorFlow.
        @tf.custom_gradient
        def differentiable_loss(amps_tf):
            # Use tf.py_function to call the wrapper.
            loss, jax_grad = tf.py_function(
                func=_get_loss_and_grad_from_jax_wrapper,
                inp=[amps_tf],
                Tout=[tf.float32, tf.complex64]
            )
            
            # The backward pass function returns the real gradient.
            def grad_fn(dy):
                # Set the shape of the gradient tensor for graph compatibility
                conjugated_jax_grad = tf.math.conj(jax_grad)
                conjugated_jax_grad.set_shape(amps_tf.shape)
                return tf.cast(dy, tf.complex64) * conjugated_jax_grad

            return loss, grad_fn
        
        losses = []
        epochs_plotted = []

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        line, = ax.plot([], [], '.-', label='Fidelity Loss')
        ax.set_yscale('log'); ax.set_xlabel('Epochs'); ax.set_ylabel('Fidelity Loss')
        ax.set_title('Live Simulation Fine-tuning Loss'); ax.grid(True, which="both"); ax.legend()

        for epoch in tqdm(range(epochs), desc="Sim-tuning"):
            with tf.GradientTape() as tape:
                raw_amps = self.model(time_input)
                generated_amps = self._apply_pulse_processing_tf(raw_amps)
                loss = differentiable_loss(generated_amps)
                # loss = differentiable_loss(generated_amps)
                # generated_amps = self._apply_pulse_processing_tf(raw_amps)
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            # Apply gradient   clipping for stability
            # clipped_grads = [(tf.clip_by_value(g, -1.0, 1.0) if g is not None else None) for g in grads]
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            epochs_plotted.append(epoch)
            losses.append(loss.numpy())
            
            # Update the plot periodically to avoid slowdown
            if epoch % update_plot_every == 0 or epoch == epochs - 1:

                line.set_xdata(epochs_plotted)
                line.set_ydata(losses)
                ax.relim(); ax.autoscale_view(); fig.canvas.draw(); fig.canvas.flush_events()
                plt.pause(0.01)
            
            if not plt.fignum_exists(fig.number):
                print("Plot window closed. Stopping fine-tuning.")
                break

            if loss.numpy() <= target_fidelity_loss:
                print(f"\nTarget fidelity loss of {target_fidelity_loss} reached at epoch {epoch}. Stopping fine-tuning.")
                break
        
        # plt.ioff()
        # plt.close(fig)

        print(f"Simulation fine-tuning finished. Final Fidelity Loss: {loss.numpy():.6f}")


    def fine_tune_hardware_spsa(self, test_loss_calculator, spsa_steps=100, a=0.01, c=0.01,
                                test_params=None, plot_live=True, update_plot_every=10,
                                target_fidelity_loss=1e-3):
        """
        Fine-tunes the final layer's biases (8 parameters) using SPSA.
        
        Args:
            test_loss_calculator (function): The loss function (hardware or sim).
            spsa_steps (int): Number of SPSA iterations.
            a (float): SPSA step size parameter.
            c (float): SPSA perturbation size parameter.
            test_params (dict, optional): If provided, passes this to the loss function.
                                          Used for testing SPSA in a simulated environment.
            plot_live (bool): Toggles live plotting of the loss.
            update_plot_every (int): How often to update the live plot.
            target_fidelity_loss (float): Target loss for early stopping.
        """

        print("\n--- Starting Hardware Fine-tuning (SPSA) ---")
        time_array = self.params['tpulse'][:-1]
        time_input = tf.constant(time_array, dtype=tf.float32)
        time_input = tf.reshape(time_input, (1, len(time_array), 1))
        
        # --- Freeze all layers except the biases of the final hidden layer ---
        for layer in self.model.layers:
            # Assumes the final hidden layer is named 'final_hidden_layer'
            if layer.name == 'final_hidden_layer':
                layer.trainable = True
                layer.kernel.trainable = False # Freeze kernel (weights)
                layer.bias.trainable = True   # Keep the 8 biases trainable
            else:
                layer.trainable = False
        target_layer = self.model.get_layer('final_hidden_layer')
        
        # --- SPSA parameters ---
        alpha, gamma = 0.602, 0.101 # Standard SPSA decay parameters

        losses = []
        epochs_plotted = []

        if plot_live:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            line, = ax.plot([], [], '.-', label='SPSA Measured Loss')
            ax.set_yscale('log'); ax.set_xlabel('SPSA Steps'); ax.set_ylabel('Fidelity Loss')
            ax.set_title('Live SPSA Hardware Tuning'); ax.grid(True, which="both"); ax.legend()

        for k in tqdm(range(spsa_steps), desc="SPSA Tuning"):

            # SPSA gain sequences that decrease over time
            ak = a / (k + 1)**alpha
            ck = c / (k + 1)**gamma

            # Get current weights (biases) and create a random perturbation vector
            theta_k = target_layer.get_weights()[1] # Get just the 8 biases
            delta_k = np.random.choice([-1, 1], size=theta_k.shape)
            
            # Create two perturbed weight sets (+ and -)
            theta_plus = theta_k + ck * delta_k
            theta_minus = theta_k - ck * delta_k
            
            # Generate two pulses and measure the loss for each
            # Measurement for the '+' perturbation
            target_layer.set_weights([target_layer.get_weights()[0], theta_plus])
            raw_amps_plus = self.model(time_input) # Keep this as a TensorFlow Tensor
            processed_amps_plus = self._apply_pulse_processing_tf(raw_amps_plus) # Process as a Tensor
            # Convert to NumPy only at the very end
            y_plus = test_loss_calculator(processed_amps_plus.numpy(), test_params) if test_params else test_loss_calculator(processed_amps_plus.numpy())

            # Measurement for the '-' perturbation
            target_layer.set_weights([target_layer.get_weights()[0], theta_minus])
            raw_amps_minus = self.model(time_input) # Keep this as a TensorFlow Tensor
            processed_amps_minus = self._apply_pulse_processing_tf(raw_amps_minus) # Process as a Tensor
            # Convert to NumPy only at the very end
            y_minus = test_loss_calculator(processed_amps_minus.numpy(), test_params) if test_params else test_loss_calculator(processed_amps_minus.numpy())

            # Approximate the gradient from only two measurements
            g_k = (y_plus - y_minus) / (2 * ck) * delta_k
            
            # Update the biases
            theta_new = theta_k - ak * g_k
            target_layer.set_weights([target_layer.get_weights()[0], theta_new])
            
            current_loss = (y_plus + y_minus) / 2 # Use the average loss for plotting
            epochs_plotted.append(k)
            losses.append(current_loss)
            
            if k % update_plot_every == 0 or k == spsa_steps - 1:
                if plot_live:
                    line.set_xdata(epochs_plotted); line.set_ydata(losses)
                    ax.relim(); ax.autoscale_view(); fig.canvas.draw(); fig.canvas.flush_events()
                    plt.pause(0.01)
                    if not plt.fignum_exists(fig.number): break
            
            if current_loss <= target_fidelity_loss:
                print(f"\nTarget fidelity loss of {target_fidelity_loss} reached at step {k}.")
                break
        
        # if plot_live: 
        #     plt.ioff()
        #     plt.close(fig)
        
        final_loss = test_loss_calculator(self.model(time_input).numpy(), test_params) if test_params else test_loss_calculator(self.model(time_input).numpy())
        print(f"Hardware fine-tuning finished. Final Measured Loss: {final_loss:.6f}")

    def fine_tune_hardware_spsa_all(self, test_loss_calculator, spsa_steps=1000, a=0.001, c=0.01,
                                    layer_name='final_hidden_layer', test_params=None, plot_live=True, 
                                    update_plot_every=10, target_fidelity_loss=1e-3):
        """
        Fine-tunes ALL parameters (weights and biases) of a target layer using SPSA.
        
        Args:
            layer_name (str): The name of the target layer to tune.
            ... (other args are the same as the previous SPSA method)
        """
        print(f"\n--- Starting Hardware Fine-tuning (SPSA on ALL weights of layer: {layer_name}) ---")
        time_array = self.params['tpulse'][:-1]
        time_input = tf.constant(time_array, dtype=tf.float32)
        time_input = tf.reshape(time_input, (1, len(time_array), 1))
        
        # --- Freeze all layers except the target layer ---
        for layer in self.model.layers:
            if layer.name == layer_name:
                layer.trainable = True
            else:
                layer.trainable = False
        target_layer = self.model.get_layer(layer_name)
        print(f"Targeting {target_layer.count_params()} parameters in layer '{layer_name}'.")

        # --- Helper logic to handle flattening/unflattening weights ---
        initial_weights_list = target_layer.get_weights()
        shapes = [w.shape for w in initial_weights_list]
        sizes = [w.size for w in initial_weights_list]
        
        def unflatten_weights(flat_theta):
            new_weights_list = []
            start = 0
            for i in range(len(shapes)):
                end = start + sizes[i]
                new_weights_list.append(flat_theta[start:end].reshape(shapes[i]))
                start = end
            return new_weights_list
        # -------------------------------------------------------------

        alpha, gamma = 0.602, 0.101
        losses, best_losses_history, epochs_plotted = [], [], []
        
        # Initialize with the current flattened weights
        theta_k = np.concatenate([w.flatten() for w in initial_weights_list])
        best_theta = np.copy(theta_k)
        initial_raw_amps = self.model(time_input)
        initial_processed_amps = self._apply_pulse_processing_tf(initial_raw_amps)
        best_loss_so_far = test_loss_calculator(initial_processed_amps.numpy(), test_params) if test_params else test_loss_calculator(initial_processed_amps.numpy())
        print(f"Initial loss before SPSA: {best_loss_so_far:.6f}")

        if plot_live:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            line, = ax.plot([], [], '.-', label='Avg. Measured Loss')
            line_best, = ax.plot([], [], 'g-', label='Best Loss So Far', linewidth=2)
            ax.set_yscale('log'); ax.set_xlabel('SPSA Steps'); ax.set_ylabel('Fidelity Loss')
            ax.set_title(f'Live SPSA Tuning ({layer_name})'); ax.grid(True, which="both"); ax.legend()

        for k in tqdm(range(spsa_steps), desc=f"SPSA Tuning ({layer_name})"):
            ak = a / (k + 1)**alpha
            ck = c / (k + 1)**gamma

            # Perturb the entire flattened vector of weights and biases
            delta_k = np.random.choice([-1, 1], size=theta_k.shape)
            theta_plus = theta_k + ck * delta_k
            theta_minus = theta_k - ck * delta_k
            
            # --- Measurement for the '+' perturbation ---
            target_layer.set_weights(unflatten_weights(theta_plus))
            raw_amps_plus = self.model(time_input)
            processed_amps_plus = self._apply_pulse_processing_tf(raw_amps_plus)
            y_plus = test_loss_calculator(processed_amps_plus.numpy(), test_params) if test_params else test_loss_calculator(processed_amps_plus.numpy())

            # --- Measurement for the '-' perturbation ---
            target_layer.set_weights(unflatten_weights(theta_minus))
            raw_amps_minus = self.model(time_input)
            processed_amps_minus = self._apply_pulse_processing_tf(raw_amps_minus)
            y_minus = test_loss_calculator(processed_amps_minus.numpy(), test_params) if test_params else test_loss_calculator(processed_amps_minus.numpy())

            # Check for best result
            if y_plus < best_loss_so_far: 
                best_loss_so_far = y_plus
                best_theta = theta_plus
            if y_minus < best_loss_so_far: 
                best_loss_so_far = y_minus
                best_theta = theta_minus
            
            # Approximate gradient and update the flattened vector
            g_k = (y_plus - y_minus) / (2 * ck) * delta_k
            theta_k = theta_k - ak * g_k
            
            current_loss = (y_plus + y_minus) / 2
            epochs_plotted.append(k)
            losses.append(current_loss)
            best_losses_history.append(best_loss_so_far)

            if k % update_plot_every == 0 or k == spsa_steps - 1:
                if plot_live:
                    line.set_xdata(epochs_plotted); line.set_ydata(losses)
                    line_best.set_xdata(epochs_plotted); line_best.set_ydata(best_losses_history)
                    ax.relim(); ax.autoscale_view(); fig.canvas.draw(); fig.canvas.flush_events()
                    plt.pause(0.01)
                    if not plt.fignum_exists(fig.number):
                        print("Plot window closed. Stopping SPSA.")
                        break
            
            if best_loss_so_far <= target_fidelity_loss:
                print(f"\nTarget fidelity loss of {target_fidelity_loss} reached at step {k}.")
                break
        
        # if plot_live:
        #     plt.ioff()
        #     plt.close(fig)
        
        # Restore the best parameters found during the run
        print(f"\nOptimization finished. Restoring best parameters found with loss: {best_loss_so_far:.6f}")
        target_layer.set_weights(unflatten_weights(best_theta))

    def fine_tune_hardware_bo(self, loss_calculator, layer_name='final_hidden_layer',
                              n_calls=100, search_radius=1.5,
                              test_params=None, plot_live=True):
        """
        Fine-tunes a target layer's parameters using Bayesian Optimization.

        Args:
            loss_calculator (function): The hardware loss function.
            layer_name (str): The name of the target layer to tune.
            n_calls (int): Number of experimental evaluations.
            search_radius (float): Defines the search space for each weight
                                   as [initial_weight +/- search_radius].
            test_params (dict, optional): Parameters for the loss function.
            plot_live (bool): Toggles live plotting of the loss.
        """
        print(f"\n--- Starting Hardware Fine-tuning (Bayesian Optimization on {layer_name}) ---")
        
        # Isolate and flatten the parameters of the target layer ---
        target_layer = self.model.get_layer(layer_name)
        initial_weights_list = target_layer.get_weights()
        shapes = [w.shape for w in initial_weights_list]
        sizes = [w.size for w in initial_weights_list]
        theta_initial = np.concatenate([w.flatten() for w in initial_weights_list])
        
        def unflatten_weights(flat_theta):
            new_weights_list = []
            start = 0
            for i, (shape, size) in enumerate(zip(shapes, sizes)):
                end = start + size
                new_weights_list.append(flat_theta[start:end].reshape(shape))
                start = end
            return new_weights_list

        # Define the search space for the optimizer ---
        # A hypercube centered around the initial pre-trained weights
        search_space = [
            Real(low=val - search_radius, high=val + search_radius, name=f"p_{i}")
            for i, val in enumerate(theta_initial)
        ]

        # Define the objective function for the optimizer ---
        # This function bridges the gap between the optimizer and the experiment
        @use_named_args(search_space)
        def objective_function(**params):
            # The optimizer provides a dictionary of named parameters
            theta_current = np.array(list(params.values()))
            
            # Update the model with the new parameters
            target_layer.set_weights(unflatten_weights(theta_current))
            
            # Generate pulse and get loss from the hardware
            pulse = self.generate_pulse()
            loss = loss_calculator(pulse, test_params) if test_params else loss_calculator(pulse)
            
            return loss

        # Setup live plotting (optional) ---
        if plot_live:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            line_best, = ax.plot([], [], 'g-', label='Best Loss Found', linewidth=2)
            ax.set_yscale('log'); ax.set_xlabel('Function Evaluations'); ax.set_ylabel('Loss')
            ax.set_title(f'Live Bayesian Optimization ({layer_name})'); ax.grid(True); ax.legend()
            
            best_losses = []
            def live_plot_callback(res):
                best_loss = np.min(res.func_vals)
                best_losses.append(best_loss)
                line_best.set_xdata(range(len(best_losses)))
                line_best.set_ydata(best_losses)
                ax.relim(); ax.autoscale_view(); fig.canvas.draw(); fig.canvas.flush_events()
                plt.pause(0.01)
                if not plt.fignum_exists(fig.number):
                    raise StopIteration("Plot window closed.")
        else:
            live_plot_callback = None

        # Run the Bayesian Optimizer ---
        try:
            result = gp_minimize(
                func=objective_function,
                dimensions=search_space,
                n_calls=n_calls,
                x0=theta_initial.tolist(), # Start with the pre-trained weights
                callback=live_plot_callback,
                acq_func="EI" # Expected Improvement is a good default
            )
        except StopIteration as e:
            print(f"\nOptimization stopped early: {e}")
            result = e

        # Set the final best parameters and report results ---
        best_params = result.x
        final_loss = result.fun
        
        print(f"\nOptimization finished. Restoring best parameters found.")
        target_layer.set_weights(unflatten_weights(np.array(best_params)))
        print(f"Final best loss: {final_loss:.6f}")



######################
# End of optimizers.py
######################