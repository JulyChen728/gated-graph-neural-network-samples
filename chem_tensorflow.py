#!/usr/bin/env/python

import json
import os
import pickle
import random
import time
from typing import List, Any, Sequence

import numpy as np

from utils import MLP, ThreadedIterator, SMALL_NUMBER

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import librispeech
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import transformer

import tensorflow.compat.v1 as tf

VOCAB_SIZE = 10

class ChemModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 100,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 0,

            'train_file': 'molecules_train.json',
            'valid_file': 'molecules_valid.json',
            
            'maxLength': 100
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)
        tb_log_dir = os.path.join(log_dir, "tb", self.run_id)
        os.makedirs(tb_log_dir, exist_ok=True)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'], is_training_data=False)

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()
            self.make_summaries()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.train_step_id, self.valid_step_id = self.restore_progress(restore_file)
            else:
                self.initialize_model()
                self.train_step_id = 0
                self.valid_step_id = 0
            self.train_writer = tf.summary.FileWriter(os.path.join(tb_log_dir, 'train'), graph=self.graph)
            self.valid_writer = tf.summary.FileWriter(os.path.join(tb_log_dir, 'validation'), graph=self.graph)

    def load_data(self, file_name, is_training_data: bool):
        print("Enter the function 'load_data' ")
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)
        print("***the data type of 'data' is "+str(type(data)))

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        #num_fwd_edge_types = 0
        #for g in data:
        #    self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
        #    num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        #self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        #self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return data

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
       
        hparams = transformer.transformer_small()
        hparams.hidden_size = 8
        hparams.filter_size = 32
        hparams.num_heads = 1
        hparams.layer_prepostprocess_dropout = 0.0
        
        if hparams.get("problem_hparams", None) is None:
            p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE,
                                                     VOCAB_SIZE,
                                                     hparams)
        hparams.problem_hparams = p_hparams
        self.model = model_cls(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
        
        self.placeholders['inputs'] = tf.placeholder(tf.int32, [None,params['maxLength']],
                                                            name='inputs')
        self.placeholders['targets'] = tf.placeholder(tf.int32, [None,params['maxLength']],
                                                            name='targets')
        features = {
            "inputs": self.placeholders['inputs'],
            "targets": self.placeholders['targets'],
            "target_space_id": 0
        }
        self.logits = self.model(features)
        
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logit=tf.reshape(self.logits, [-1, VOCAB_SIZE]),
            labels=tf.reshape(features["targets"], [-1]))
        self.ops['loss'] = tf.reduce_mean(loss)
        apply_grad = tf.train.AdamOptimizer(0.001).minimize(self.ops['loss'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                           self.placeholders['out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                                                                                self.placeholders['out_layer_dropout_keep_prob'])
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id])
                diff = computed_values - self.placeholders['target_values'][internal_id, :]
                task_target_mask = self.placeholders['target_mask'][internal_id, :]
                task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
                diff = diff * task_target_mask  # Mask out unused values
                self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
                task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num
                # Normalise loss to account for fewer task-specific examples in batch:
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['losses'].append(task_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        #optimize the loss
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        
        # Initialize newly-introduced variables:
        #self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def make_summaries(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.ops['loss'])
            for task_id in self.params['task_ids']:
                tf.summary.scalar('accuracy%i' % task_id, self.ops['accuracy_task%i' % task_id])
        self.ops['summary'] = tf.summary.merge_all()

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, is_training: bool, start_step: int = 0):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_data = 0
        steps = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        
        with tf.Session() as session:
            
            for my_step, my_batch_data in enumerate(batch_iterator):
                precessed_data += self.params['batch_size']
                #other information should be in fetch_list like loss
                fetch_list = [self.ops['loss'], logits]
                result = session.run(fetch_list,feed_dict = my_batch_data)
                #result is the output layer
                (batch_loss, batch_logits) = (result[0],result[1])
                loss += batch_loss * self.params['batch_size'] 
                print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                               step,
                                                                               self.params['batch_size'],
                                                                               loss / processed_data),
                  end='\r')
                steps += 1
            loss = loss / processed_data
            instance_per_sec = processed_data / (time.time() - start_time)
        return loss, instance_per_sec, steps 
        
        #for step, batch_data in enumerate(batch_iterator):
        #    num_graphs = batch_data[self.placeholders['num_graphs']]
        #    processed_graphs += num_graphs
        #    if is_training:
        #        batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
        #        fetch_list = [self.ops['loss'], accuracy_ops, self.ops['summary'], self.ops['train_step']]
        #    else:
        #        batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
        #        fetch_list = [self.ops['loss'], accuracy_ops, self.ops['summary']]
            
            #fetch_list = [input_data, target_data]
            #my_result = self.sess.run(fetch_list,feed_dict = batch_data)
            #output = self.sess.run(logits)
            
        #    result = self.sess.run(fetch_list, feed_dict=batch_data)
        #    (batch_loss, batch_accuracies, batch_summary) = (result[0], result[1], result[2])
        #    writer = self.train_writer if is_training else self.valid_writer
        #    writer.add_summary(batch_summary, start_step + step)
        #    loss += batch_loss * num_graphs
        #    accuracies.append(np.array(batch_accuracies) * num_graphs)

        #    print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
        #                                                                       step,
        #                                                                       num_graphs,
        #                                                                       loss / processed_graphs),
        #          end='\r')
        #    steps += 1

        #accuracies = np.sum(accuracies, axis=0) / processed_graphs
        #loss = loss / processed_graphs
        #error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        #instance_per_sec = processed_graphs / (time.time() - start_time)
        #return loss, accuracies, error_ratios, instance_per_sec, steps

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                _, valid_accs, _, _, steps = self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_speed, train_steps = self.run_epoch("epoch %i (training)" % epoch,
                                                                                              self.train_data, True, self.train_step_id)
                #train_loss, train_accs, train_errs, train_speed, train_steps = self.run_epoch("epoch %i (training)" % epoch,
                #                                                                              self.train_data, True, self.train_step_id)
                self.train_step_id += train_steps
                #accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                #errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | instances/sec: %.2f" % (train_loss,
                                                                                                       # accs_str,
                                                                                                       # errs_str,
                                                                                                        train_speed))
                valid_loss, valid_speed, valid_steps = self.run_epoch("epoch %i (validation)" % epoch,
                                                                                              self.valid_data, False, self.valid_step_id)
                #valid_loss, valid_accs, valid_errs, valid_speed, valid_steps = self.run_epoch("epoch %i (validation)" % epoch,
                #                                                                              self.valid_data, False, self.valid_step_id)
                self.valid_step_id += valid_steps
                #accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                #errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | instances/sec: %.2f" % (valid_loss,
                                                                                                        #accs_str,
                                                                                                        #errs_str,
                                                                                                        valid_speed))

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_speed),
                    'valid_results': (valid_loss, valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)
                    
                #accuracy is needed
                val_acc = np.sum(valid_accs)  # type: float
                if val_acc < best_val_acc:
                    self.save_progress(self.best_model_file, self.train_step_id, self.valid_step_id)
                    print("  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" % (
                        val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    break

    def save_progress(self, model_path: str, train_step: int, valid_step: int) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save,
            "train_step": train_step,
            "valid_step": valid_step,
        }

        with open(model_path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_progress(self, model_path: str) -> (int, int):
        print("Restoring weights from file %s." % model_path)
        with open(model_path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

        return data_to_load['train_step'], data_to_load['valid_step']
