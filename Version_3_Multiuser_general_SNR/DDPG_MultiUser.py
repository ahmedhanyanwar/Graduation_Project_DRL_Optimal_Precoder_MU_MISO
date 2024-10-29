import os
from keras.optimizers import Adam
import tensorflow as tf

from buffer_numpy import ReplayBuffer
from networks_MultiUser import ActorNetwork, CriticNetwork

class Agent:
    """
    A class to represent an agent using the Deep Deterministic Policy Gradient (DDPG) algorithm.
    """
    
    def __init__(self, input_dims, alpha=0.001, beta=0.002,
                 gamma=1, n_users=4, n_tx=10, max_size=1000000, tau=.005,
                 fc1=512, fc2=512, fc3=512, batch_size=64, noise=0.1,
                 changing_alg='LIN', model_comment='', chkpt_dir='ddpg_weights/model_'):
        """Initializes the Agent with the specified parameters."""
        # Initialize hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.tau = tau  # Soft update coefficient for target networks
        self.n_users = n_users  # Number of users in the multi-user scenario
        self.n_actions = 2 * n_users * n_tx  # Total number of actions (real and imaginary components)
        self.batch_size = batch_size  # Size of the batch for training
        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions, self.batch_size)  # Replay buffer
        self.noise = noise  # Standard deviation for action noise
        self.alpha = alpha  # Learning rate for the actor network
        self.beta = beta  # Learning rate for the critic network

        self.__comment = model_comment  # Comment or description for the model
        self.model_num = changing_alg  # Identifier for the model version
        self.model_path = chkpt_dir + changing_alg  # Path to save/load model weights
        self.check_path(self.model_path)  # Check if the directory exists

        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(fc1, fc2, fc3, n_users, n_tx)  # Actor network
        self.critic = CriticNetwork(fc1, fc2, fc3)  # Critic network

        # Initialize target networks
        self.target_actor = ActorNetwork(fc1, fc2, fc3, n_users, n_tx, name='target_actor')
        self.target_critic = CriticNetwork(fc1, fc2, fc3, name='target_critic')

        # Compile networks with optimizers
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        # Hard copy initial weights of actor and critic to target networks
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        """Updates the weights of target networks using soft update."""
        if tau is None:
            tau = self.tau  # Use the instance's tau if not provided

        # Soft update for Actor
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)  # Set updated weights for target actor

        # Soft update for Critic
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)  # Set updated weights for target critic

    def remember(self, state, action, reward, next_state):
        """Stores state, action, reward, and next state in the buffer."""
        self.memory.store_transition(state, action, reward, next_state)

    def save_models(self, name=''):
        """Saves the models' weights to the specified path."""
        print(f'____Saving model of ({self.model_num}) version {name}____')
        self.actor.save_weights(os.path.join(self.model_path, 'actor_' + name + '.h5'))
        self.critic.save_weights(os.path.join(self.model_path, 'critic_' + name + '.h5'))
        self.target_actor.save_weights(os.path.join(self.model_path, 'target_actor_' + name + '.h5'))
        self.target_critic.save_weights(os.path.join(self.model_path, 'target_critic_' + name + '.h5'))

    def load_models(self, name=''):
        """Loads the models' weights from the specified path."""
        print(f'____Loading model of ({self.model_num}) version {name}____')
        self.actor.load_weights(os.path.join(self.model_path, 'actor_' + name + '.h5'))
        self.critic.load_weights(os.path.join(self.model_path, 'critic_' + name + '.h5'))
        self.target_actor.load_weights(os.path.join(self.model_path, 'target_actor_' + name + '.h5'))
        self.target_critic.load_weights(os.path.join(self.model_path, 'target_critic_' + name + '.h5'))

    def choose_action(self, state, evaluate=False):
        """Chooses an action based on the current state."""
        states = tf.convert_to_tensor([state])  # Convert state to tensor
        actions = self.actor(states)  # Get action from actor network

        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)  # Add noise to actions
            actions_mat = actions  # Action matrix
            actions_mat = self.actor.normalize_layer(actions_mat)  # Normalize actions
            actions = tf.reshape(actions_mat, (actions.shape[0], -1))  # Reshape actions
        return actions

    def learn(self, is_end_episode=0.0, evaluate=False):
        """Updates the agent's knowledge based on experience."""
        if self.memory.mem_cntr < self.batch_size:  # Check if enough samples in memory
            if not evaluate:
                return
            else:
                state, action, reward, next_state = self.memory.load_first()  # Load first sample
        else:
            state, action, reward, next_state = self.memory.sample_buffer()  # Sample from memory

        # Convert states and actions to tensors
        states = tf.convert_to_tensor(state)
        next_states = tf.convert_to_tensor(next_state)
        actions = tf.convert_to_tensor(action)
        rewards = tf.convert_to_tensor(reward)

        # Calculate gradients for critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)  # Get actions from target actor
            next_critic_value = tf.squeeze(self.target_critic(next_states, target_actions), 1)  # Evaluate target actions
            critic_value = tf.squeeze(self.critic(state, actions), 1)  # Evaluate current actions

            # Calculate target using rewards and next critic value
            target = rewards + (self.gamma * next_critic_value) * (1 - is_end_episode)  # Target for critic
            critic_loss = tf.keras.losses.MSE(target, critic_value)  # Mean squared error loss

        # Calculate and apply gradients for critic network
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        # Calculate gradients for actor network
        with tf.GradientTape() as tape:
            next_actions = self.actor(states)  # Get actions from current actor
            actor_loss = -self.critic(states, next_actions)  # Actor loss
            actor_loss = tf.math.reduce_mean(actor_loss)  # Mean loss

        # Calculate and apply gradients for actor network
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        # Update target networks
        self.update_network_parameters()

    def check_path(self, chkpt_dir):
        """Checks if the checkpoint directory exists, creates it if not."""
        if not (os.path.isdir(os.path.split(chkpt_dir)[0])):
            os.mkdir(os.path.split(chkpt_dir)[0])  # Create base directory if it doesn't exist
        if not (os.path.isdir(chkpt_dir)):
            os.mkdir(chkpt_dir)  # Create checkpoint directory if it doesn't exist

    def set_comment(self, comment):
        """Writes a comment to a text file in the model path."""
        with open(self.model_path + "/CommentFile.txt", "a") as comment_file:
            comment_file.write(comment + "\n")  # Append comment

    def get_comment(self):
        """Reads and prints the comment from the text file."""
        with open(self.model_path + "/CommentFile.txt", "r") as read_comment:
            print(read_comment.read())  # Print the comment
