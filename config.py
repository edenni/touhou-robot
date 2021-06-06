n_actions = 18
n_stack = 4

# memory capacity
capacity = 1000

batch_size = 16
gamma = 0.999
eps_train = 0.4
eps_train_final = 0.01

roi = ('main', 'manshinsoui', 'score')

screen_size = (960, 1080, 3)
image_size = (84, 84)
state_shape = (1, 84, 84)
action_shape = 18
hidden_sizes = 512

max_grad_norm = 0.5