import numpy as np


class ReplayBuffer():

    def __init__( self, max_size, input_shape, n_actions ):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros( ( self.mem_size, input_shape ) )
        self.action_memory = np.zeros( ( self.mem_size, n_actions ) )
        self.new_state_memory = np.zeros( ( self.mem_size, input_shape) )
        self.reward_memroy = np.zeros( self.mem_size )
        self.terminal_memory = np.zeros( self.mem_size, dtype=np.bool )

    def store_transition( self, state, action, new_state, reward, done ):
        index = self.mem_cntr % self.mem_size
        self.state_memory[ index ] = state
        self.action_memory[ index ] = action
        self.new_state_memory[ index ] = new_state 
        self.reward_memroy[ index ] = reward
        self.terminal_memory[ index ] = done
        self.mem_cntr += 1

    def sample_buffer( self, batch_size ):
        max_mem = min( self.mem_cntr, self.mem_size )

        batch = np.random.choice( max_mem, batch_size )
        states = self.state_memory[ batch ]
        actions = self.action_memory[ batch ]
        new_states = self.new_state_memory[ batch ]
        rewards = self.reward_memroy[ batch ]
        dones = self.terminal_memory[ batch ]

        return states, actions, new_states, rewards, dones