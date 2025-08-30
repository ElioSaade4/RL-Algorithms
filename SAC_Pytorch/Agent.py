import os
import numpy as np
import torch as T
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
from CriticNetwork import CriticNetwork
from ValueNetwork import ValueNetwork
from ActorNetwork import ActorNetwork


class Agent():
    
    def __init__( self, alpha=0.0003, beta=0.0003, input_dims = 8, env=None, gamma=0.99, n_actions=2, 
                  max_size=1000000, tau=0.05, layer1_size=256, batch_size=256, reward_scale=2 ):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer( max_size, input_dims, n_actions )
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork( alpha, input_dims, max_action=env.action_space.high,
                                   n_actions=n_actions, name='actror' )
        
        self.critic_1 = CriticNetwork( beta, input_dims, n_actions=n_actions, name='critic_1' )
        self.critic_2 = CriticNetwork( beta, input_dims, n_actions=n_actions, name='critic_2' )

        self.value = ValueNetwork( beta, input_dims, name='value' )
        self.target_value = ValueNetwork( beta, input_dims, name='target_value' )

        self.scale = reward_scale
        self.update_network_parameters( tau=1 )

    
    def choose_action( self, observation ):
        state = T.tensor( [ observation ], dtype=T.float ).to( self.actor.device )
        actions, _ = self.actor.sample_normal( state, reparametrize=False )
        
        return actions.cpu().detach().numpy()[ 0 ]
    

    def remember( self, state, action, new_state, reward, done ):
        self.memory.store_transition( state, action, new_state, reward, done )


    def learn( self ):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        # Sample the replay buffer and convert to CUDA tensors
        state, action, new_state, reward, done = self.memory.sample_buffer( self.batch_size )
        state  = T.tensor( state, dtype=T.float ).to( self.actor.device )
        action = T.tensor( action, dtype=T.float ).to( self.actor.device )
        new_state = T.tensor( new_state, dtype=T.float ).to( self.actor.device )
        reward = T.tensor( reward, dtype=T.float).to( self.actor.device )
        done = T.tensor( done ).to( self.actor.device )

        # Calculate needed values for Value network update
        # view( -1 ) will basically return a single tensor for the whole batch. 
        # Basically, the value for every sample in the batch will be a tensor with 1 element,
        # so we will have a tensor of tensors [[], [], []], which we don't need
        value = self.value.forward( state ).view( -1 )
        value_ = self.target_value.forward( new_state ).view( -1 )    # WHY new_state??
        value_[ done ] = 0.0  # value of terminal state is 0

        actions, log_probs = self.actor.sample_normal( state, reparametrize=False )
        log_probs = log_probs.view( -1 )
        
        q1_new_policy = self.critic_1.forward( state, actions )
        q2_new_policy = self.critic_2.forward( state, actions )
        critic_value = T.min( q1_new_policy, q2_new_policy )
        critic_value = critic_value.view( -1 )

        # Calculate loss for Value network and optimize
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss( value, value_target )
        value_loss.backward( retain_graph=True )
        self.value.optimizer.step()

        # Calculate needed values for Actor network update
        actions, log_probs = self.actor.sample_normal( state, reparametrize=True )
        log_probs = log_probs.view( -1 )
        q1_new_policy = self.critic_1.forward( state, actions )
        q2_new_policy = self.critic_2.forward( state, actions )
        critic_value = T.min( q1_new_policy, q2_new_policy )
        critic_value = critic_value.view( -1 )

        # Calculate loss for Actor network and optimize
        self.actor.optimizer.zero_grad()
        actor_loss = log_probs - critic_value
        actor_loss = T.mean( actor_loss )
        actor_loss.backward( retain_graph=True )
        self.actor.optimizer.step()

        # Critic networks update
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward( state, action ).view( -1 )
        q2_old_policy = self.critic_2.forward( state, action ).view( -1 )
        critic_1_loss = 0.5 * F.mse_loss( q1_old_policy, q_hat )
        critic_2_loss = 0.5 * F.mse_loss( q2_old_policy, q_hat )

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update target network paramaters
        self.update_network_parameters()


    def update_network_parameters( self, tau=None ):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict( target_value_params )
        value_state_dict = dict( value_params )

        for name in value_state_dict:
            value_state_dict[ name ] = tau * value_state_dict[ name ].clone() + \
                                        ( 1 - tau ) * target_value_state_dict[ name ].clone()
            
        self.target_value.load_state_dict( value_state_dict )

    
    def save_models( self ):
        print( '... saving models ...' )
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()


    def load_models( self ):
        print( '... loading models ...' )
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()