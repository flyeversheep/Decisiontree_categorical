import numpy as np
class Node:
    def __init__(self, label = None):

        # The names of all possible y labels
        self.label_names = None
        
        # The majority y label of the current node
        self.label = label
        
        # The feature name used at this node
        self.feature = None
        
        # The gini value at this node
        self.gini = None
        
        # The value of the feature at this node
        self.feature_value = None
        
        # The number of different y labels at this node
        self.value = None
        
        # Used to record the feature value of the lower level
        # and the correponding node
        self.lower_level = {}

def gini_value(Y):
    '''Calculate the Gini value of sample Y 
    Parameters
    ----------
    Y: a Pandas Series, length n
        The Series containing different labels.
    Returns
    -------
    gini : int
        A gini value of Y.
    '''
    ratios = Y.value_counts() / sum(Y.value_counts())
    
    gini = 1 - np.sum(np.square(ratios))
    
    return gini

def select_best_feature_multi_split(X, Y, candidates, criterion = 'gini'):
    '''Select the best feature 
    Parameters
    ----------
    X: a Pandas DataFrame, shape n by m
        The DataFrame containing features.
    Y: a Pandas Series, length n
        The Series containing different labels.
    candidates: list, leng k
        The list containing all candidate features
    criterion: string
        The criterion used to select the best feature.
    Returns
    -------
    min_feature: string
        The name of the best feature selected.
    min_gini: float
        The Gini index from the selected split.
    ''' 
    
    n = X.shape[0]
    
    if criterion == 'gini':
        
        min_gini = float('inf')
        min_feature = None
        
        for feature in candidates:
            
            gini_index = 0
            
            for value in X[feature].unique():
                subset = X[feature] == value
                gini_subset = gini_value(Y[subset])
                gini_index += sum(subset) * gini_subset / n
            
            if gini_index < min_gini:
                min_gini = gini_index
                min_feature = feature
        
        return min_feature, min_gini
            
def select_best_feature_binary_split(X, Y, candidates, responds_names, criterion = 'gini'):
    '''Select the best feature and perform binary split, can only be used in binary classification 
       task. 
    Parameters
    ----------
    X: a Pandas DataFrame, shape n by m
        The DataFrame containing features.
    Y: a Pandas Series, length n
        The Series containing different labels.
    candidates: list, length k
        The list containing all candidate features.
    responds_names: list, length 2
        The possible responds names.
    criterion: string
        The criterion used to select the best feature.
    Returns
    -------
    min_feature: string
        The name of feature selected
    best_split: (2,) tuple of list
        ([list of feature value larger than threshold], 
         [list of feature value not larger than threshold])
    min_gini: float
        The Gini index from the selected split. 
    '''
    n = X.shape[0]
    
    if criterion == 'gini':
        
        min_gini = float('inf')
        min_feature = None
        best_split = None
        
        for feature in candidates:
            
            ratios = []
            # Dictionary used to store the number of subsamples and the ratio of label 1 vs 0
            subsample_num = {}
            
            for feature_value in X[feature].unique():
                # subset that equals to a specific feature value
                subset = X[feature] == feature_value
                # count the number of different labels and store it in an array
                count_temp = [sum(Y[subset] == responds_names[i]) for i in range(len(responds_names))]
                # Calculate the ratio of label
                ratio_temp =  count_temp / sum(count_temp)
                # 
                #subsample_num[ratio_temp[0]] = (count_temp, feature_value)
                subsample_num[feature_value] = (count_temp, ratio_temp[0])
                # Store the ratio of the first label
                ratios.append(ratio_temp[0])
            
            ratios = sorted(ratios)
            
            for split in ratios[:-1]:
                
                # not_larger and larger value is used to store the
                # cumulated value for two classes
                not_larger = [0, 0]
                larger = [0, 0]
                
                # Lists used to store feature names based on different split
                larger_feature_value = []
                not_larger_feature_value = []
                
                for test_feature in subsample_num:
                    test_count, test_ratio = subsample_num[test_feature]
                
                    if test_ratio > split:
                        
                        larger = [larger[i] + test_count[i] for i in range(2)]
                        larger_feature_value.append(test_feature)
                    else:
                        not_larger = [not_larger[i] + test_count[i] for i in range(2)]
                        not_larger_feature_value.append(test_feature)
                
                if sum(larger) == 0:
                    continue
                else:
                    gini_larger = 1 - (larger[0]/sum(larger)) ** 2 - (larger[1]/sum(larger)) ** 2
                
                if sum(not_larger) == 0:
                    continue
                else:
                    gini_not_larger = 1 - (not_larger[0]/sum(not_larger)) ** 2\
                                        - (not_larger[1]/sum(not_larger)) ** 2
                
                # calculate the current gini_index based on the current split.
                gini_index = (gini_larger * sum(larger) + gini_not_larger * sum(not_larger))\
                                / (sum(larger) + sum(not_larger))
                
                if gini_index < min_gini:
                    min_feature = feature
                    min_gini = gini_index
                    best_split = (larger_feature_value, not_larger_feature_value)
                
        return min_feature, best_split, min_gini
        
def TreeGenerate(X, Y, features,
                 criterion = 'gini', 
                 current_feature = None, 
                 current_feature_value = None,
                 split = 'multiway',
                 min_split = 25,
                 gini_decrease_min = 0.1):
    '''Generate a decision tree. 
    Parameters
    ----------
    X: a Pandas DataFrame, shape n by m
        The DataFrame containing features.
    Y: a Pandas Series, length n
        The Series containing different labels.
    features: list, leng k
        The list containing all candidate features.
    criterion: string
        The criterion used to select the best feature.
    current_feature: string
        The key of the feature at current split.
    current_feature_value: string
        The value of the current feature at current node.
    split: string
        The option to split each node: 'mutiway', 'binary'
    min_split: int
        The minimum sample number in order to perform a split
    gini_decrease_min: float
        The minimum decrease in gini index in order to perform the split
    
    Returns
    -------
    current_node : Node object
        The current node.
    ''' 
    # The respond column name of Y
    respond = Y.columns[0]
    
    # Create a new node as the current node
    current_node = Node()
    
    # Calculate the majority label of Y
    current_node.label = Y.loc[:, respond].value_counts().idxmax()
    
    # Initialize the label_names attribute of the current node
    current_node.label_names = sorted(list(Y[respond].unique()))
    
    # Calculate and store other attributes of the current node
    current_node.gini = gini_value(Y.loc[:, respond])
    current_node.feature = current_feature
    current_node.feature_value = current_feature_value
    current_node.value = [sum(Y[respond] == i) for i in current_node.label_names]
    
    # if there is no feature, the samples have the same value for
    # all features or all the sample belong to the same category, exit the recursion
    if len(features) == 0\
        or all(X[features].nunique() <= 1)\
        or Y.loc[:, respond].nunique() == 1:
        return current_node
    
    # if there are not enough samples to conduct a split
    if Y.shape[0] < min_split:
        return current_node
    
    if split == 'multiway':
        # Select the best feature based on multiway splitting
        best_feature, min_gini = select_best_feature_multi_split(X, Y.loc[:, respond], features, criterion)

        if current_node.gini - min_gini < gini_decrease_min:
            return current_node
        
        new_features = list(features)
        new_features.remove(best_feature)

        # Multiway spliting at the node and recursively calculate 
        # the children nodes.
        for feature_value in X[best_feature].unique():

            subset = X[best_feature] == feature_value
            X_subset = X.loc[subset, :]
            Y_subset = Y.loc[subset, :]

            if X_subset.shape[0] == 0:
                lower_node = Node(Y_subset.value_counts().idxmax())
            else:
                lower_node = TreeGenerate(X_subset, 
                                          Y_subset, 
                                          new_features,
                                          criterion, 
                                          best_feature,
                                          feature_value,
                                          split,
                                          min_split,
                                          gini_decrease_min)

            current_node.lower_level[feature_value] = lower_node
    
    elif split == 'binary':
        # Select the best feature based on binary splitting
        best_feature, best_split, min_gini = select_best_feature_binary_split(X, 
                                                                              Y.loc[:, respond], 
                                                                              features,  
                                                                              current_node.label_names,
                                                                              criterion)
        # if there is not enough improvement from the split
        if current_node.gini - min_gini < gini_decrease_min:
            return current_node

        new_features = list(features)
        new_features.remove(best_feature)
        
        for i in range(2):
            subset = [item in best_split[i] for item in X[best_feature]]
            X_subset = X[subset]
            Y_subset = Y[subset]
            if X_subset.shape[0] == 0:
                lower_node = Node(majority(Y_subset[0]))
            else:
                lower_node = TreeGenerate(X_subset, 
                                          Y_subset, 
                                          new_features,
                                          criterion, 
                                          best_feature,
                                          best_split[i],
                                          split,
                                          min_split,
                                          gini_decrease_min)
            current_node.lower_level[i] = lower_node
    
    return current_node

# borrowed from sklearn
def _color_brew(n):
    """Generate n colors with equally spaced hues.
    Parameters
    ----------
    n : int
        The number of colors required.
    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list



def export_graph(node, file_name, filled = False):
    
    """
    Export a decision tree in DOT format.
    This function generates a GraphViz representation of the decision tree,
    which is then written into 'file_name'. Once exported, graphical renderings
    can be generated using, for example::
        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)
    Parameters
    ----------
    node : Node object
        The root node.
    file_name: String
        The file name of the dot file.
    filled: boolean
        Whether to fill the tree node with color
    Returns
    -------
    """
    # borrowed from sklearn
    def get_color(value):
        # Find the appropriate color & intensity for a node
        # ONLY work for Classification tree
        color = list(colors[np.argmax(value)])
        sorted_values = sorted(value, reverse=True)
        if len(sorted_values) == 1:
            alpha = 0
        else:
            alpha = int(np.round(255 * (sorted_values[0] -
                                        sorted_values[1]) /
                                       (1 - sorted_values[1]), 0))
    
        # Return html color code in #RRGGBBAA format
        color.append(alpha)
        hex_codes = [str(i) for i in range(10)]
        hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
        color = [hex_codes[c // 16] + hex_codes[c % 16] for c in color]

        return '#' + ''.join(color)
    
    def recursive(node, out_file, filled, node_id = 0):
        """
        Recursively write the file to out_file.
        Parameters
        ----------
        node : Node object
            The current node.
        out_file: TextIOWrapper
            The textIOwrapper to which we write the file.
        filled: boolean
            Whether to fill the tree node with color.
        node_id: int
            ID of the current node.
        Returns
        -------
        node_id: int
            ID of the next node.
        """
        out_file.write('%d [label="' %node_id)
    
        if node_id == 0:
            # root node
            out_file.write('Root\\n')
            
        # Write down the different attributes of the node. 
        elif node.feature is not None:
            out_file.write('%s : %s\\n' %(node.feature, node.feature_value))

        out_file.write('gini = %f\\n' %node.gini)
        out_file.write('samples = %d\\n' %sum(node.value))
        out_file.write('value = %s\\n' % node.value)
        out_file.write('class = %s' % node.label)
        out_file.write('"')
        
        # Color fill.
        if filled:
            node_value = node.value / sum(node.value)
            out_file.write(',fillcolor="%s"' % get_color(node_value))

        out_file.write('] ;\n')
        current_node_id = node_id
        node_id += 1
        for next_node in node.lower_level:
            new_node_id = recursive(node.lower_level[next_node], out_file, filled, node_id)
            out_file.write('%d -> %d ;\n'%(current_node_id, node_id))
            node_id = new_node_id
        return node_id
    
    
    # Start of export_graph
    out_file = open(file_name, "w", encoding="utf-8")
    
    # Write the header 
    out_file.write('digraph Tree {\n')   
    out_file.write('node [shape=box')
    
    if filled:
        # Write the color related line and brew colors based on the 
        # number of label values.
        out_file.write(', style="filled", color="black"')
        colors = _color_brew(len(node.value))
    
    out_file.write('] ;\n')
    
    # Recursively render each node
    recursive(node, out_file, filled)
    
    out_file.write("}")
    

    