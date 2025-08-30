import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ValueNetwork( nn.Module ):

    def __init__( self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac' ):
        # Parent class constructor
        super( ValueNetwork, self ).__init__()

        # Save some attributes
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join( self.checkpoint_dir, name + '_sac' )

        # Create neural network
        self.fc1 = nn.Linear( self.input_dims, self.fc1_dims )
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims )
        self.v = nn.Linear( self.fc2_dims, 1 )

        self.optimizer = optim.Adam( self.parameters(), lr=beta )
        self.device = T.device( 'cuda:0' if T.cuda.is_available() else 'cpu' )

        self.to( self.device )


    def forward( self, state ):
        state_value = F.relu( self.fc1( state ) )
        state_value = F.relu( self.fc2( state_value ) )
        state_value = self.v( state_value )

        return state_value
    

    def save_checkpoint( self ):
        print( '... saving checkpoint ...' )
        T.save( self.state_dict(), self.checkpoint_file )


    def load_checkpoint( self ):
        print( '... loading checkpoint ...' )
        self.load_state_dict( T.load( self.checkpoint_file ) )  