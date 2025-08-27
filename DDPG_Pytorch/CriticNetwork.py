import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork( nn.Module ):
    # derive the network from the class nn.Module in order to be able to use the train and eval functions, 
    # and self.parameters

    def __init__( self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="tmp/ddpg" ):
        """
        beta: learning rate
        input_dims: shape of the input (state)
        fc1_dims, shape of the first fully connected layer
        fc2_dims: shape of the second fully connected layer
        n_actions: number of actions to be taken by the agent
        name: name for the NN, used for saving it
        chkpt_dir: directory for saving
        """
        super( CriticNetwork, self ).__init__()
        self.learning_rate = beta       # I added this
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join( chkpt_dir, name + '_ddpg' )

        self.fc1 = nn.Linear( *self.input_dims, fc1_dims )
        f1 = 1 / np.sqrt( self.fc1.weight.data.size()[ 0 ] )   # number to initialize the weights and biases of fc1
        # this is to constraint the initial weights of the network to a narrow range of values to help with convergence
        T.nn.init.uniform_( self.fc1.weight.data, -f1, f1 )
        T.nn.init.uniform_( self.fc1.bias.data, -f1, f1 )
        self.bn1 = nn.LayerNorm( self.fc1_dims )        # batch normalization, helps with convergence

        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims )
        f2 = 1 / np.sqrt( self.fc2.weight.data.size()[ 0 ] )
        T.nn.init.uniform_( self.fc2.weight.data, -f2, f2 )
        T.nn.init.uniform_( self.fc2.bias.data, -f2, f2 )
        self.bn2 = nn.LayerNorm( self.fc2_dims )
        # the fact that we have batch normalization layer means that we have to use the eval and train functions later

        self.action_value = nn.Linear( self.n_actions, self.fc2_dims )
        
        f3 = 0.003
        self.q  = nn.Linear( self.fc2_dims, 1 )
        T.nn.init.uniform_( self.q.weight.data, -f3, f3 )
        T.nn.init.uniform_( self.q.bias.data, -f3, f3 )

        self.optimizer = optim.Adam( self.parameters(), lr= self.learning_rate )
        self.device = T.device( 'cuda' if T.cuda.is_available() else 'cpu' )

        self.to( self.device )


    def forward( self, state, action ):
        state_value = self.fc1( state )
        state_value = self.bn1( state_value )
        state_value = F.relu( state_value )
        # it's an open debate whether we relu before batch normalization or after it

        state_value = self.fc2( state_value )
        state_value = self.bn2( state_value )

        # action_value below is getting double relu activated. 
        # We can play around with this because relu is not commutative with addition
        action_value = self.action_value( action )
        action_value = F.relu( action_value )

        state_action_value = F.relu( T.add( state_value, action_value ) )
        state_action_value = self.q( state_action_value )

        return state_action_value
    

    def save_checkpoint( self ):
        print( '... saving chekpoint ...' )
        T.save( self.state_dict(),  self.checkpoint_file )
        # self.dict() is from the parent class, it returns a dict where the keys are the parameter names
        # and the values are the values are the parameter values


    def load_checkpoint( self ):
        print( '... loading checkpoint ...' )
        self.load_state_dict( T.load( self.checkpoint_file ) )