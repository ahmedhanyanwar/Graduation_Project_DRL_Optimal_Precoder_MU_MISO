import numpy as np

class ReplayBuffer:
    """A class to implement a replay buffer for storing experiences in reinforcement learning."""
    def __init__(self, max_size, input_shape, n_actions, batch_size=64):
        """
        Initializes the replay buffer.

        Parameters:
        - max_size: Maximum number of experiences to store in the buffer.
        - input_shape: Shape of the state input (e.g., dimensions of state).
        - n_actions: Number of possible actions.
        - batch_size: Size of the batch to sample during training (default is 64).
        """
        self.mem_size = max_size  # Set the maximum size of the buffer.
        self.mem_cntr = 0          # Initialize the counter for stored experiences.
        
        # Initialize arrays to store state, next state, action, and reward experiences.
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.batch_size = batch_size  # Set the batch size for sampling.

    def store_transition(self, state, action, reward, next_state):
        """
        Stores a transition (state, action, reward, next_state) in the replay buffer.

        Parameters:
        - state: The current state of the environment.
        - action: The action taken by the agent.
        - reward: The reward received after taking the action.
        - next_state: The next state of the environment after the action.
        """
        index = self.mem_cntr % self.mem_size  # Calculate index for circular storage.
        
        # Store the transition in the respective memory arrays.
        self.state_memory[index]      = state
        self.next_state_memory[index] = next_state
        self.action_memory[index]     = action
        self.reward_memory[index]     = reward

        self.mem_cntr += 1  # Increment the memory counter.

    def sample_buffer(self):
        """
        Randomly samples a batch of transitions from the replay buffer.

        Returns:
        - state_batch: Batch of states sampled from the buffer.
        - action_batch: Batch of actions sampled from the buffer.
        - reward_batch: Batch of rewards sampled from the buffer.
        - next_state_batch: Batch of next states sampled from the buffer.
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # Determine the maximum number of stored experiences.
        
        # Randomly sample indices from the stored experiences.
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # Retrieve the corresponding batches from the memory arrays.
        state_batch      = self.state_memory[batch]
        next_state_batch = self.next_state_memory[batch]
        action_batch     = self.action_memory[batch]
        reward_batch     = self.reward_memory[batch]
        
        # Return the sampled batch of (state, action, reward, next_state).
        return state_batch, action_batch, reward_batch, next_state_batch
    
    def load_first(self):
        """
        Generates a batch of random initial transitions.

        Returns:
        - state_batch: Randomly generated batch of states.
        - action_batch: Randomly generated batch of actions.
        - reward_batch: Randomly generated batch of rewards.
        - next_state_batch: Randomly generated batch of next states.
        """
        # Generate random batches for state, next state, action, and reward.
        state_batch = np.random.random((self.batch_size, self.state_memory.shape[1])).astype(np.float32)
        next_state_batch = np.random.random((self.batch_size, self.next_state_memory.shape[1])).astype(np.float32)
        action_batch = np.random.random((self.batch_size, self.action_memory.shape[1])).astype(np.float32)
        reward_batch = np.random.random((self.batch_size,)).astype(np.float32)

        # Return the randomly generated batch of (state, action, reward, next_state).
        return state_batch, action_batch, reward_batch, next_state_batch