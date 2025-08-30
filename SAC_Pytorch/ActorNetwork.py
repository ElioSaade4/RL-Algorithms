import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class ActorNetwork( nn.Module ):

    def __init__( self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor',
                  chkpt_dir='tmp/sac' ):
        # Parent class constructor
        super( ActorNetwork, self ).__init__()

        # Save some attributes
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join( self.checkpoint_dir, name + '_sac' )
        self.reparam_noise = 1e-6

        # Create the neural netowrk
        self.fc1 = nn.Linear( self.input_dims, self.fc1_dims )
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims )
        self.mu = nn.Linear( self.fc2_dims, self.n_actions )
        self.sigma = nn.Linear( self.fc2_dims, self.n_actions )

        self.optimizer = optim.Adam( self.parameters(), lr=alpha )
        self.device = T.device( 'cuda:0' if T.cuda.is_available() else 'cpu' )

        self.to( self.device )


    def forward( self, state ):
        l1 = self.fc1( state )
        act_l1 = F.relu( l1 )
        l2 = self.fc2( act_l1 )
        act_l2 = F.relu( l2 )

        # print( f'l1 = {l1}, act_l1 = {act_l1}, l2 = {l2}, act_l2 = {act_l2}' )

        mu = self.mu( act_l2 )
        sigma = self.sigma( act_l2 )

        # we clip sigma in order to limit the width of the probability distribution
        # we use the repram_noise so that we do not have a std dev of 0
        # we could have used a sigmoid activation function for sigma [0,1], but it is slower computationally
        sigma = T.clamp( sigma, min=self.reparam_noise, max=1 )

        return mu, sigma
    

    def sample_normal( self, state, reparametrize=True ):
        mu, sigma = self.forward( state )
        probabilities = Normal( mu, sigma )

        if reparametrize:
            actions = probabilities.rsample()  # adds some exploration noise
        else:
            actions = probabilities.sample()

        # this comes from the Appendix of the paper
        # log_probs is used for the loss calculation in the learning of the neural network
        action = T.tanh( actions ) * T.tensor( self.max_action ).to( self.device )
        log_probs = probabilities.log_prob( actions )

        # + self.reparam_noise so that we don't have 0 in the log
        log_probs -= T.log( 1 - action.pow( 2 ) + self.reparam_noise )

        # we take the sum because we need a scalar quantity for loss, whereas log_probs will have the same dimension as n_actions
        log_probs = log_probs.sum( 1, keepdim=True )
        
        return action, log_probs
    

    def save_checkpoint( self ):
        print( '... saving checkpoint ...' )
        T.save( self.state_dict(), self.checkpoint_file )


    def load_checkpoint( self ):
        print( '... loading checkpoint ...' )
        self.load_state_dict( T.load( self.checkpoint_file ) )  