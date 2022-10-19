from rectipy import Network, random_connectivity, input_connections, wta_score
import numpy as np
import pickle
import sys


# preparations
##############

# network parameters
N = 1000
p = 0.1
eta = 0.0
Delta = 0.1
etas = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
v_theta = 1e3
Delta_in = 2.0
J = 7.0  #sys.argv[-1]

# input parameters
m = 5
p_in = 0.1
s1 = [0, 2, 1]
s2 = [4, 2, 3]
signals = [s1, s2]

# output parameters
k = len(signals)

# training parameters
T_init = 50.0
T_syll = 0.5
n_syll = len(s1)
n_reps = 50
T_epoch = T_syll*n_syll*n_reps
dt = 1e-3
n_epochs = 5
train_epochs = 4

# define extrinsic input and targets
epoch_steps = int(T_epoch/dt)
syll_steps = int(T_syll/dt)
init_steps = int(T_init/dt)
inp = np.zeros((n_epochs, epoch_steps, m))
targets = np.zeros((n_epochs, epoch_steps, k))
for epoch in range(n_epochs):
    for rep in range(n_reps):
        choice = np.random.choice(2)
        s = signals[choice]
        for idx in range(n_syll):
            inp[epoch, (rep*n_syll+idx)*syll_steps:(rep*n_syll+idx+1)*syll_steps, s[idx]] = 1.0
        targets[epoch, rep*n_syll*syll_steps:(rep+1)*n_syll*syll_steps, choice] = 1.0

# optimization
##############

n_runs = 2
scores = []
for i in range(n_runs):

    # generate connectivity matrix
    W = random_connectivity(N, N, p, normalize=True)

    # generate input matrix
    W_in = input_connections(N, m, p_in, variance=Delta_in, zero_mean=True)

    # initialize network
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.qif.qif_sfa_pop", weights=W*J,
                            source_var="s", target_var="s_in", input_var_ext="I_ext", output_var="s", spike_var="v",
                            input_var_net="spike", op="qif_sfa_op", node_vars={'all/qif_sfa_op/eta': etas}, dt=dt,
                            spike_threshold=v_theta, spike_reset=-v_theta, float_precision="float64", record_vars=['s'],
                            clear=True)

    # wash out initial condition
    net.run(np.zeros((init_steps, 1)), verbose=False, sampling_steps=init_steps+1)

    # add input and output layers
    net.add_input_layer(N, m, weights=W_in, trainable=False)
    net.add_output_layer(N, k, trainable=True, activation_function='softmax')
    net.compile()

    # perform optimization
    net.train(inp[:train_epochs], targets[:train_epochs], optimizer='sgd', loss='ce', sampling_steps=1,
              optimizer_kwargs={'momentum': 0.1, 'dampening': 0.9}, lr=1e-2)

    # test performance on last epoch
    obs, loss = net.test(inp[train_epochs], targets[train_epochs], loss='ce', record_output=True, sampling_steps=1,
                         verbose=False)

    # calculate WTA score
    wta = wta_score(np.asarray(obs['out']), targets[train_epochs])
    scores.append(wta)
    print(f'Finished run #{i}. Loss on test data set: {loss}. WTA score: {wta}.')

pickle.dump(scores, open('data/qif_sfa_syns_opt.pkl', 'wb'))
