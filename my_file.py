#!/usr/bin/env/python

from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json

from chem_tensorflow import ChemModel
from utils import glorot_init, SMALL_NUMBER

dir = '/home/qian/gzip-1.10/input/'

class MYTRANSFORMERModel(ChemModel):

    def __init__(self, args):
        super().__init__(args)
        
    
    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 10,
            'use_edge_bias': False,
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                                     "2": [0],
                                     "4": [0, 2]
                                    },

            'layer_timesteps': [2, 2, 1, 2, 1],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'edge_weight_dropout_keep_prob': .8
        })
        return params
    
    def make_minibatch_iterator(self, data: Any, is_training: bool):
        if is_training:
            np.random.shuffle(data)
        num_sample = 0
        
        while num_sample < len(data):
            num_offset = 0
            inputs = []
            targets = []
            while num_sample < len(data) and num_offset < self.params['batch_size']:
                d = data[num_sample]
                for key in d:
                    #key is the file name
                    file_full_name = dir + key
                    #read in the file content in binary format
                    with open(file_full_name,"rb") as fp:
                        input = fp.read() 
                        count = len(input)%32
                        if count!=0:
                            #add 0
                            input = '0'*(32-count) + input
                        input_vertor = [] 
                        i = 0
                        while i < len(input):
                            temps = input[i:i+32]
                            tempi = int(temps,2)
                            input_vector.append(tempi)
                            i = i + 32  
                        inputs.append(input_vertor)
                    target = d[key]
                    targets.append(target)
                    num_sample += 1
                    num_offset += 1
            batch_feed_dict = {
                self.placeholders['inputs']: np.array(inputs),
                self.placeholders['targets']: np.array(targets)
            }
           
            yield batch_feed_dict
            
            
        assert num_sample == len(data)
        
 def main():
    args = docopt(__doc__)
    try:
        model = MYTRANSFORMERModel(args)
        if args['--evaluate']:
            model.example_evaluation()
        else:
            model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()       
