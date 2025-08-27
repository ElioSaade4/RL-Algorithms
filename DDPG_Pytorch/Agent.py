import torch as T
import torch.nn.functional as F
import numpy as np

from OUActionNoise import OUActionNoise
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetowrk
from CriticNetwork import CriticNetwork


class Agent( object ):

    def __init__( self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000,
                 layer1_size=400, layer2_size=300, batch_size=64 ):
        """
        alpha: learning rate of actor network
        beta: learning rate of critic network
        input_dims: number of dimensions of the state
        tau: constant for polyak averaging to update the parameters of the target networks
        env: the environment in which the agent interacts
        gamma: discount factor
        n_actions: number of dimensions of the actions
        max_size: size of the replay buffer
        layer1_size: number of neurons in first layer
        layer2_size: number of neurons in the second layer
        batch_size: size of the batch sampled from the replay buffer
        """
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayBuffer( max_size, input_dims, n_actions )
        self.batch_size = batch_size

        self.actor = ActorNetowrk( alpha, input_dims, layer1_size, layer2_size, n_actions, 'Actor' )
        self.target_actor = ActorNetowrk( alpha, input_dims, layer1_size, layer2_size, n_actions, 'TargetActor' )
        self.critic = CriticNetwork( beta, input_dims, layer1_size, layer2_size, n_actions, 'Critic' )
        self.target_critic = CriticNetwork( beta, input_dims, layer1_size, layer2_size, n_actions, 'TargetCritic' )

        self.noise = OUActionNoise( mu=np.zeros( n_actions ) )

        self.update_network_parameters( tau=1 ) 
        # set tau to 1 here so that the parameters of the actor & critic get copied to the target networks


    def choose_action( self, observation ):
        # eval does not perform an evaluation step. It puts the actor in evaulation mode
        # it tells python that you don't want to calculate statistics for the batch normalization
        # Very Important! If you don't do that, the agent will not learn
        # Same for train function, it puts it in training mode but does store statistics for batch normalization
        # if you don't do batch normalization, then eval and train not needed, but they are needed if we do dropout
        self.actor.eval()
        
        observation = T.tensor( observation, dtype=T.float ).to( self.actor.device )

        # I added .forward()
        action = self.actor.forward( observation ).to( self.actor.device )
        action_prime = action + T.tensor( self.noise(), dtype=T.float ).to( self.actor.device )

        self.actor.train()

        return action_prime.cpu().detach().numpy() # bring it back to numpy value because we cannot pass a tensor to openAI gym
    

    def remember( self, state, action, reward, new_state, done ):
        self.memory.store_transition( state, action, reward, new_state, done )


    def learn( self ):
        if self.memory.mem_cntr < self.batch_size:
            return
        else:

            # Randomly sample a batch
            state, action, reward, new_state, done = self.memory.sample_buffer( self.batch_size )
            #pdb.set_trace()
            state = T.tensor( state, dtype=T.float ).to( self.critic.device )
            action = T.tensor( action, dtype=T.float ).to( self.critic.device )
            reward = T.tensor( reward, dtype=T.float ).to( self.critic.device )
            new_state = T.tensor( new_state, dtype=T.float ).to( self.critic.device )
            done = T.tensor( done ).to( self.critic.device )

            # Compute targets
            self.target_actor.eval()
            self.target_critic.eval()
            self.critic.eval()

            target_actions = self.target_actor.forward( new_state )
            target_critic_value = self.target_critic.forward( new_state, target_actions )
            critic_values = self.critic.forward( state, action )

            target = [0] * self.batch_size
            for j in range( self.batch_size ):
                target[ j ] = reward[ j ] + ( self.gamma * target_critic_value[ j ] * done[ j ] )

            target = T.tensor( target ).to( self.critic.device )
            target = target.view( self.batch_size, 1 )  # to reshape

            # Calculate the critic loss and train the crtitic
            self.critic.train()
            self.critic.optimizer.zero_grad()
            # in pytorch, we have to zero the gradients every time so that they do not accumulate and interfere with the calculations
            critic_loss = F.mse_loss( target, critic_values )
            critic_loss.backward()
            self.critic.optimizer.step()

            # Calculate the actor loss and train the actor
            self.critic.eval()
            self.actor.train()
            self.actor.optimizer.zero_grad()
            mu = self.actor.forward( state )
            actor_loss = - self.critic.forward( state, mu )
            actor_loss = T.mean( actor_loss )
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the target actor and target critic parameters
            self.update_network_parameters()


    def update_network_parameters( self, tau=None ):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_dict = dict( actor_params )
        critic_dict = dict( critic_params )
        target_actor_dict = dict( target_actor_params )
        target_critic_dict = dict( target_critic_params )
        
        # Update target actor parameters
        for name in actor_dict:
            target_actor_dict[ name ] = tau * actor_dict[ name ].clone() + \
                                         ( 1 - tau ) * target_actor_dict[ name ].clone()
            
        self.target_actor.load_state_dict( target_actor_dict )
        
        # Update target critic parameters
        for name in critic_dict:
            target_critic_dict[ name ] = tau * critic_dict[ name ].clone() + \
                                         ( 1 - tau ) * target_critic_dict[ name ].clone()
            
        self.target_critic.load_state_dict( target_critic_dict )


    def save_models( self ):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()


    def load_models( self ):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()