import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import warnings

def neighborhood_sampling(x, x_label, closest_counterfactual, 
                          cut_radius=False, best_threshold=None, 
                          kind="uniform_sphere", verbose=False, 
                          n=1000, n_batch=100, forced_balance_ratio=None, 
                          balance=True, apply_bb_predict=None, **kwargs):

    # Utilizar el contrafactual más cercano proporcionado
    if closest_counterfactual is None:
        raise ValueError("Please provide a closest_counterfactual value.")
    
    # Calcular el radio si es necesario
    if cut_radius:
        best_threshold = np.linalg.norm(x - closest_counterfactual)
        if verbose:
            print("Setting new threshold at radius:", best_threshold)
        if kind not in ["uniform_sphere"]:
            warnings.warn("cut_radius=True, but for the method " + kind + " the threshold is not a radius.")
    
    # Aquí es donde deberíamos definir la función `vicinity_sampling`
    Z = vicinity_sampling(closest_counterfactual.reshape(1,-1), n=n, threshold=best_threshold, **kwargs)

    # Si se proporciona un ratio de balanceo forzado, entonces intenta equilibrar el vecindario
    if forced_balance_ratio is not None:
        y = apply_bb_predict(Z)
        y = 1 * (y == x_label)
        
        n_minority_instances = np.sum(y == 0)  # Asumiendo que 0 es la clase minoritaria
        if (n_minority_instances / n) < forced_balance_ratio:
            if verbose:
                print("Forced balancing neighborhood...", end=" ")
            n_desired_minority_instances = int(forced_balance_ratio * n)
            n_desired_majority_instances = n - n_desired_minority_instances
            
            # Aquí es donde deberíamos definir la función `vicinity_sampling` de nuevo.
            while n_minority_instances < n_desired_minority_instances:
                Z_ = vicinity_sampling(closest_counterfactual.reshape(1,-1), n=n_batch, threshold=best_threshold, **kwargs)
                
                y_ = apply_bb_predict(Z_)
                y_ = 1 * (y_ == x_label)
                
                n_minority_instances += np.sum(y_ == 0)  # Asumiendo que 0 es la clase minoritaria
                
                Z = np.vstack([Z, Z_])
                y = np.concatenate([y, y_])
            
            rus = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
            Z, y = rus.fit_resample(Z, y)
            if len(Z) > n:
                Z, _ = train_test_split(Z, train_size=n, stratify=y)
            if verbose:
                print("Done!")
    
    # Equilibrar el vecindario si es necesario
    if balance:
        if verbose:
            print("Balancing neighborhood...", end=" ")
        rus = RandomUnderSampler(random_state=0)
        y = apply_bb_predict(Z)
        y = 1 * (y == x_label)
        Z, _ = rus.fit_resample(Z, y)
        if verbose:
            print("Done!")
    
    return Z

def vicinity_sampling(x, kind, n, threshold=None, verbose=False, **kwargs):

    if verbose:
        print("\nSampling -->", kind)
    
    if kind == "gaussian":
        Z = gaussian_vicinity_sampling(x, threshold, n)
    elif kind == "gaussian_matched":
        # Z = gaussian_matched_vicinity_sampling(x, threshold, n)
        raise NotImplementedError("gaussian_matched_vicinity_sampling is not defined yet.")
    elif kind == "gaussian_global":
        Z = gaussian_global_sampling(x, n)
    elif kind == "uniform_sphere":
        Z = uniform_sphere_vicinity_sampling(x, n, threshold)
    elif kind == "uniform_sphere_scaled":
        Z = uniform_sphere_scaled_vicinity_sampling(x, n, threshold)
    else:
        raise Exception("Vicinity sampling kind not valid", kind)
    
    return Z


def gaussian_vicinity_sampling(z, epsilon, n=1):
    return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)

def gaussian_global_sampling(z, n=1):
    return np.random.normal(size=(n, z.shape[1]))

# La función uniform_sphere_origin ya está bien definida, así que la dejamos como está.

def uniform_sphere_origin(n, d, r=1):
    """Generate "num_points" random points in "dimension" that have uniform probability over the unit ball scaled
    by "radius" (length of points are in range [0, "radius"]).

    Parameters
    ----------
    n : int
        number of points to generate
    d : int
        dimensionality of each point
    r : float
        radius of the sphere

    Returns
    -------
    array of shape (n, d)
        sampled points
    """
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(d, n))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(n) ** (1 / d)
    # Return the list of random (direction & length) points.
    return r * (random_directions * random_radii).T

def uniform_sphere_vicinity_sampling(z, n=1, r=1):
    Z = uniform_sphere_origin(n, z.shape[1], r)
    return translate(Z, z)

def uniform_sphere_scaled_vicinity_sampling(z, n=1, threshold=1):
    Z = uniform_sphere_origin(n, z.shape[1], r=1)
    Z *= threshold
    return translate(Z, z)

def translate(X, center):
    """
    Translates a origin centered array to a new center.

    Parameters
    ----------
    X : array
        data to translate centered in the axis origin
    center : array
        new center point

    Returns
    -------
    array
        Translated data.
    """
    for axis in range(center.shape[-1]):
        X[..., axis] += center[..., axis]
    return X
