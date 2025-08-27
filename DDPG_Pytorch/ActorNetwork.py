import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActorNetowrk( nn.Module ):

    def __init__( self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg' ):
        super( ActorNetowrk, self ).__init__()
        self.learning_rate = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join( chkpt_dir, name + '_ddpg' )

        self.fc1 = nn.Linear( *self.input_dims, self.fc1_dims )
        f1 = 1 / np.sqrt( self.fc1.weight.data.size()[ 0 ] )
        T.nn.init.uniform_( self.fc1.weight.data, -f1, f1 )
        T.nn.init.uniform_( self.fc1.bias.data, -f1, f1 )
        
        self.bn1 = nn.LayerNorm( self.fc1_dims )

        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims )
        f2 = 1 / np.sqrt( self.fc2.weight.data.size()[ 0 ] )
        T.nn.init.uniform_( self.fc2.weight.data, -f2, f2 )
        T.nn.init.uniform_( self.fc2.bias.data, -f2, f2 )

        self.bn2 = nn.LayerNorm( self.fc2_dims )

        f3 = 0.003
        self.mu = nn.Linear( self.fc2_dims, self.n_actions )
        T.nn.init.uniform_( self.mu.weight.data, -f3, f3 )
        T.nn.init.uniform_( self.mu.bias.data, -f3, f3 )

        self.optimizer = optim.Adam( self.parameters(), lr= self.learning_rate )
        self.device = T.device( 'cuda' if T.cuda.is_available() else 'cpu' )
        self.to( self.device )


    def forward( self, state ):
        action = self.fc1( state )
        action = self.bn1( action )
        action = F.relu( action )
        action = self.fc2( action )
        action = self.bn2( action )
        action = F.relu( action )
        action = T.tanh( self.mu( action ) )
        # tanh will bound it between -1 and 1, we can then multiply it by the actual bounds of the action

        return action
    

    def save_checkpoint( self ):
        print( '... saving chekpoint ...' )
        T.save( self.state_dict(),  self.checkpoint_file )
        # self.dict() is from the parent class, it returns a dict where the keys are the parameter names
        # and the values are the values are the parameter values


    def load_checkpoint( self ):
        print( '... loading checkpoint ...' )
        self.load_state_dict( T.load( self.checkpoint_file ) )