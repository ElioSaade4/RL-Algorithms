import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class CriticNetwork( nn.Module ):

    def __init__( self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3' ):
        super( CriticNetwork, self ).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_action = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join( self.checkpoint_dir, name + '_td3' )

        self.fc1 = nn.Linear( self.input_dims + n_actions, self.fc1_dims )
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims )
        self.q1 = nn.Linear( self.fc2_dims, 1 )  # output 1 because single Q value

        self.optimizer = optim.Adam( self.parameters(), lr=beta )
        self.device = T.device( 'cuda' if T.cuda.is_available() else 'cpu' )

        self.to( self.device )

    def forward( self, state, action ):
        q1_action_value = self.fc1( T.cat( [ state, action ], dim=1 ) )
        q1_action_value = F.relu( q1_action_value )
        q1_action_value = self.fc2( q1_action_value )
        q1_action_value = F.relu( q1_action_value )
        q1_action_value = self.q1( q1_action_value )

        return q1_action_value

    def save_checkpoint( self ):
        print( '... saving checkpoint ...' )
        T.save( self.state_dict(), self.checkpoint_file )

    def load_checkpoint( self ):
        print( '... loading checkpoint ...' )
        self.load_state_dict( T.load( self.checkpoint_file ) )    