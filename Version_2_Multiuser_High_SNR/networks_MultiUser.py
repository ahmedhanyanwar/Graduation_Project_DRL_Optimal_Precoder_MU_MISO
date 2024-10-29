import tensorflow as tf
import keras
from keras.layers import Dense
from keras import backend as K
from keras.layers import Lambda

def castum_normalize_layer(x):
    """Normalizes the input tensor using L2 normalization."""
    return K.l2_normalize(x, axis=2)

class CriticNetwork(keras.Model):
    """A neural network model representing the Critic in the DDPG algorithm."""
    
    def __init__(self, fc1_dims=512, fc2_dims=512, fc3_dims=512, name='critic'):
        """Initializes the Critic network with the specified layer sizes."""
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims  # Size of the first fully connected layer
        self.fc2_dims = fc2_dims  # Size of the second fully connected layer
        self.fc3_dims = fc3_dims  # Size of the third fully connected layer

        self.model_name = name  # Model name

        # Define the layers
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.q = Dense(1, activation=None)  # Output layer for Q-value

    def call(self, states, actions):
        """Calculates the Q-value for the given states and actions."""
        # Adjust actions shape if necessary
        if len(actions.shape) > len(states.shape):
            actions = tf.squeeze(actions, axis=0)

        # Concatenate states and actions, then pass through the network
        action_value = self.fc1(tf.concat([states, actions], axis=1))
        action_value = self.fc2(action_value)
        action_value = self.fc3(action_value)

        q = self.q(action_value)  # Output Q-value
        return q


class ActorNetwork(keras.Model):
    """A neural network model representing the Actor in the DDPG algorithm."""
    
    def __init__(self, fc1_dims=512, fc2_dims=512, fc3_dims=512, n_users=4, n_tx=3, name='actor'):
        """Initializes the Actor network with the specified layer sizes and user/tx parameters."""
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims  # Size of the first fully connected layer
        self.fc2_dims = fc2_dims  # Size of the second fully connected layer
        self.fc3_dims = fc3_dims  # Size of the third fully connected layer
        self.n_users = n_users  # Number of users
        self.n_actions = 2 * n_users * n_tx  # Total actions (real and imaginary)

        self.model_name = name  # Model name

        # Define the layers
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)  # Output layer for actions
        self.normalize_layer = Lambda(castum_normalize_layer)  # Normalization layer

    def call(self, states):
        """Computes the normalized action probabilities based on the input states."""
        # Pass the input states through fully connected layer
        policy = self.fc1(states)
        policy = self.fc2(policy)
        policy = self.fc3(policy)
        # Output layer for actions
        mu = self.mu(policy)
        # If mu has four dimensions, squeeze it to reduce dimensions
        if len(mu.shape) == 4:
            mu = tf.squeeze(mu, axis=0)
        # Reshape mu to a matrix of size (#users x #antennas)
        mu_mat = tf.reshape(mu, (mu.shape[0], self.n_users, -1))
        # Normalize each row of the matrix representing W1, W2, ..., Wn (where n is the number of users)
        mu_norm_mat = self.normalize_layer(mu_mat)
        # Reshape the normalized matrix back to a vector of size (1, n_actions)
        mu_norm = tf.reshape(mu_norm_mat, (mu.shape[0], -1))
        # Return the normalized action vector
        return mu_norm

