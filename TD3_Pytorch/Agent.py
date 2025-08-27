import numpy as np
import torch as T
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
from CriticNetwork import CriticNetwork
from ActorNetwork import ActorNetwork


class Agent():

    def __init__( self, alpha, beta, input_dims, tau, env, gamma=0.99, update_actor_interval=2,
                  warmup=1000, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=30,
                  batch_size=100, noise=0.1 ):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer( max_size, input_dims, n_actions )
        self.batch_size = batch_size
        self.learn_step_cntr = 0    # because we need to delay learning
        self.time_step = 0    # to know when the warmup period has expired
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork( alpha, input_dims, layer1_size, layer2_size, n_actions, 
                                  name='actor' )
        self.critic_1 = CriticNetwork( beta, input_dims, layer1_size, layer2_size, n_actions, 
                                      name='critic_1')        
        self.critic_2 = CriticNetwork( beta, input_dims, layer1_size, layer2_size, n_actions, 
                                      name='critic_2')        
        self.target_actor = ActorNetwork( alpha, input_dims, layer1_size, layer2_size, n_actions, 
                                         name='target_actor' )
        self.target_critic_1 = CriticNetwork( beta, input_dims, layer1_size, layer2_size, n_actions, 
                                      name='target_critic_1')        
        self.target_critic_2 = CriticNetwork( beta, input_dims, layer1_size, layer2_size, n_actions, 
                                      name='target_critic_2')

        self.noise = noise
        self.update_network_parameters( tau=1 )


    def choose_action( self, observation ):
        # during warmup period, select random action from normal noise
        if self.time_step < self.warmup:
            mu = T.tensor( np.random.normal( scale=self.noise, size=( self.n_actions, ) ) ).to( self.actor.device )
        else:
            state = T.tensor( observation, dtype=T.float ).to( self.actor.device )
            mu = self.actor.forward( state ).to( self.actor.device )

        # it doesn't matter that we are adding noise to noise during the warmup period because it is
        # completely exploratory
        mu_prime = mu + T.tensor( np.random.normal( scale=self.noise ), dtype=T.float ).to( self.actor.device )

        mu_prime = T.clamp( mu_prime, self.min_action[ 0 ], self.max_action[ 0 ] )
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()
    
    def remember( self, state, action, new_state, reward, done ):
        self.memory.store_transition( state, action, new_state, reward, done )

    def learn( self ):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, new_state, reward, done = self.memory.sample_buffer( self.batch_size )

        # it doesn't matter if it's the device of actor or critic_1 because they are the same GPU
        # we use T.float as the CUDA data type because it's 32 bits instead of 64 bits
        # otherwise, it "kinda barks at you" ~Phil
        state = T.tensor( state, dtype=T.float ).to( self.critic_1.device )
        action = T.tensor( action, dtype=T.float ).to( self.critic_1.device )
        reward = T.tensor( reward, dtype=T.float ).to( self.critic_1.device )
        new_state = T.tensor( new_state, dtype=T.float ).to( self.critic_1.device )
        done = T.tensor( done ).to( self.critic_1.device )

        target_actions = self.target_actor.forward( new_state )
        # target policy smoothing
        target_actions = target_actions + \
                            T.clamp( T.tensor( np.random.normal( scale=0.2 ) ), -0.5, 0.5 )
        target_actions = T.clamp( target_actions, self.min_action[ 0 ], self.max_action[ 0 ] )

        q1_target = self.target_critic_1.forward( new_state, target_actions )
        q2_target = self.target_critic_2.forward( new_state, target_actions )

        q1 = self.critic_1.forward( state, action )
        q2 = self.critic_2.forward( state, action )

        # same as multiplying by (1-d) in the target equations
        q1_target[ done ] = 0.0
        q2_target[ done ] = 0.0

        q1_target = q1_target.view( -1 )
        q2_target = q2_target.view( -1 )

        critic_value_target = T.min( q1_target, q2_target )
        target = reward + self.gamma * critic_value_target
        target = target.view( self.batch_size, 1 )

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss( target, q1 )
        q2_loss = F.mse_loss( target, q2 )

        # Why are we adding the losses? Shouldn't each loss be used to train its own network?
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward( state, self.actor.forward( state ) )
        actor_loss = -T.mean( actor_q1_loss )
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters( self, tau=None ):
        # polyak averaging update
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor = dict( actor_params )
        critic_1 = dict( critic_1_params )
        critic_2 = dict( critic_2_params )
        target_actor = dict( target_actor_params )
        target_critic_1 = dict( target_critic_1_params )
        target_critic_2 = dict( target_critic_2_params )

        for name in actor:
            actor[ name ] = tau * actor[ name ].clone() + ( 1 - tau ) * target_actor[ name ].clone()

        for name in critic_1:
            critic_1[ name ] = tau * critic_1[ name ].clone() + ( 1 - tau ) * target_critic_1[ name ].clone()

        for name in critic_2:
            critic_2[ name ] = tau * critic_2[ name ].clone() + ( 1 - tau ) * target_critic_2[ name ].clone()

        self.target_actor.load_state_dict( actor )
        self.target_critic_1.load_state_dict( critic_1 )
        self.target_critic_2.load_state_dict( critic_2 )

    def save_models( self ):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models( self ):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()