def add_neighbor(graph_dict,neuron_id,neighbor_n_id,neighbor_i_id,modifiable=True,weight=1.0):
    """
    Adding a neighbor to the graph dictionary
    
    graph_dict - graph dictionary, the keys are the neurons and the values are lists
    neuron_id - from this neuron
    neighbor_n_id - to this neuron
    neighbor_i_id - to this input
    weight = fix weight
    modifiable = if False, then immutable
    """
    if neuron_id not in graph_dict.keys():
        graph_dict = add_neuron(graph_dict,neuron_id)
    
    if neighbor_n_id not in graph_dict.keys():
        graph_dict = add_neuron(graph_dict,neighbor_n_id)
    
       
    neighbor = None
    if modifiable == True:
        neighbor = (neighbor_n_id,neighbor_i_id,True)
    else:
        neighbor = (neighbor_n_id,neighbor_i_id,False,weight)
        
    graph_dict[neuron_id].append(neighbor)
    
    return graph_dict

def add_input(activations_dict,neuron_id,input_id,act_type,modifiable=True,weight=0.0):
    """
    Adding an input to the activations dictionary
    
    activations_dict - activations dictionary, the keys are the neurons and the values are dictionaries with the inputs
    neuron_id - the neuron
    input_id - its input
    act_type - activation type (number), we convert it to string
    weight = fix weight
    modifiable = if False, then immutable
    """
    act_type = str(act_type)
    if neuron_id in activations_dict.keys():
        if modifiable == True:
            activations_dict[neuron_id][input_id] = [ act_type, True ]
        else:
            activations_dict[neuron_id][input_id] = [ act_type, False, weight ]
    else:
        if modifiable == True:
            activations_dict[neuron_id] = {input_id : [act_type, True]}
        else:
            activations_dict[neuron_id] = {input_id : [act_type, False, weight]}
    return activations_dict

def add_neuron(graph_dict,neuron_id):
    """
    Adding neuron to graph dictionary
    
    graph_dict - graph dictionary, the keys are the neurons and the values are lists
    neuron_id - the neuron
    
    """
    if neuron_id not in graph_dict.keys():
        graph_dict[neuron_id] = [ ]
    return graph_dict
















