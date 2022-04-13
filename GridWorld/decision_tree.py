"""
the leave-one-out-cross-validation (LOOCV) classification error rate for both the training and test folds.
k − 1 = 11 data points (training folds) and predict on the 1 left out data point (test fold).
information gain to split tree

Entropy = -sum_by_category[P(xi)logP(xi)]
Information Gain (IG) = Entropy(target_column) - sum()


"""

import pandas as pd
import numpy as np

# import data
table = pd.read_csv("hw3_dataset.csv", header=None)
header_list = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait']
table.columns = header_list

# sanity check that things imported well
print(table)

# using information gain, which relies on entropy
def entropy(attribute):
    if isinstance(attribute, pd.Series):
        p = attribute.value_counts()/attribute.shape[0]
        p += 0.000001
        entropy = np.sum(-p*np.log2(p))
    return entropy

# calculation for infomration gain
def inf_gain(attribute, split_on):
    sum_mask = sum(split_on)
    sum_not_mask = split_on.shape[0] - sum_mask
    total = sum_mask + sum_not_mask

    information_gain = entropy(attribute) - sum_mask / total * entropy(attribute[split_on]) - sum_not_mask / total * entropy(attribute[-split_on])

    return information_gain

# iterate through combination types to find the best split based on the information gains
def max_ig_split(split_attribute, target):
    split_on_options = []
    information_gains = []
    # print("split_attribute" + str(split_attribute))
    combos = split_options(split_attribute)

    for split in combos:
        split_on = split_attribute.isin(list(split))
        information_gain = inf_gain(target, split_on)
        information_gains.append(information_gain)
        split_on_options.append(split)

    # print(information_gains)
    # print(split_on_options)
    if information_gains:
        return max(information_gains), split_on_options[np.argmax(information_gains)]
    else:
        return 0.0, ['None']


# need to iterate through splits
def split_options(categories):
    categories = categories.unique()
    list_of_options = []

    for combo_size in range(0, len(categories)+1):
        temp_combos = generate_combinations(categories, combo_size)
        for temp_combo in temp_combos:
            list_of_options.append(list(temp_combo))
    # print("list_of_options")
    # print(list_of_options)
    return list_of_options[1:-1]

# need all combinations to iterate through
def generate_combinations(categories, length):
    if length == 0:
        return [[]]
    
    list_of_combinations = []
    for i in range(0, len(categories)):
        category = categories[i]
        remaining_categories = categories[i + 1:]
        
        for p in generate_combinations(remaining_categories, length-1):
            list_of_combinations.append([category]+p)
             
    return list_of_combinations

# split the table against the attribute/variable that maximizes the information gain
def split_decision(target, table):
    split_on_df = table.drop(target, axis = 'columns')
    split_on_df = split_on_df.apply(max_ig_split, target = table[target])
    split_on_df = split_on_df.set_axis([0,1], axis = 'index')
    # split_on_df = split_on_df.loc[:,split_on_df.loc[0,:]]
    # print('here')
    # print(split_on_df)
    # print(split_on_df.loc[0,:])
    # print(sum(split_on_df.loc[0,:]))

    if sum(split_on_df.loc[0,:]) < 0.00000000000001:
        return None, None, None, None, 0.0
    else:
        ag = np.argmax(split_on_df.values,axis=1)
        s=pd.Series(split_on_df.columns[np.argmax(split_on_df.values,axis=1)])
        best_split = s[0]
        best_split_entropy = split_on_df[best_split][0]
        best_split_variable = split_on_df[best_split][1]

        split_cat = table[table[best_split].isin(best_split_variable)]
        alt_cats = table[table[best_split].isin(best_split_variable) == False]

    return (split_cat, alt_cats, best_split, best_split_variable, best_split_entropy)

# Max count classification against target column
def classify(table):
    classification = table.value_counts().idxmax()
    return classification

# Confusion matrix
def performance_tracking(split_cat_table, alt_cat_table, split_cat_yes, split_cat_no, target, tf_count):
    perf_count = split_cat_table[target].value_counts()
    for count,index in enumerate(list(perf_count.index)):
        if split_cat_yes == index:
            tf_count['TP'] = (tf_count.get('TP') + perf_count[count])
        elif split_cat_no == index:
            tf_count['FP'] = (tf_count.get('FP') + perf_count[count])

    perf_count = alt_cat_table[target].value_counts()
    for count,index in enumerate(list(perf_count.index)):
        if split_cat_yes == index:
            tf_count['FN'] = (tf_count.get('FN') + perf_count[count])
        elif split_cat_no == index:
            tf_count['TN'] = (tf_count.get('TN') + perf_count[count])

    return tf_count

# def final_performance(perf):
#     norm_factor=1.0/sum(perf.dict.values())
#     for key in perf:
#         perf[key] = perf[key]*norm_factor

def training(target, table, depth, tf_count, iter = 0):
    # print('training tf_count: ' + str(tf_count))
    if depth == iter:
        classification = classify(table[target])
        return classification
    else:
        split_cat, alt_cats, best_split, best_split_variable, best_split_entropy = split_decision(target, table)
        # perf = performance_tracking(split_cat, alt_cats, target)
        # print("best_split_entropy: " + str(best_split_entropy))
        # print('split_cat: ' + str(split_cat))
        # print('alt_cats: ' + str(alt_cats))
        # print('best_split: ' + str(best_split))
        # print('best_split_var: ' + str(best_split_variable))
        if best_split_entropy >= 0.00000000000001: # min information_gain
            tree_key = str(best_split + ' ' + best_split_variable[0])
            tree = {tree_key: []}
            iter += 1
            # print("split_cat: " + str(split_cat))
            # print("alt_cats: " + str(alt_cats))
            split_cat_yes = training(target, split_cat, depth, tf_count, iter)
            split_cat_no = training(target, alt_cats, depth, tf_count, iter)
            tf_count = performance_tracking(split_cat, alt_cats, split_cat_yes, split_cat_no, target, tf_count)
            # print('returned tf_count: ' + str(tf_count))
            # perf = performance(split_cat[target], alt_cats[target])
            # print(len(split_cat_yes))
            tree[tree_key].append([split_cat_yes])
            tree[tree_key].append([split_cat_no])
            # print(tree)
        else:
            classification = classify(table[target])
            return classification
        
        return tree, tf_count

def testing(test_table, tree):
    """ 
    for key, value in dictionary_tree
    if column(key[0] first word) value == key[1](key second word) then prediction = value
    """
# last minute rush to get output... it's 11:25 now and I've spent probably 10 hours on this. That's what I get for doing a master's with no BA

depth = 2
target = 'WillWait'
total_tf_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
for row in range(0,table.shape[0]):
    train_table = table.drop([row], axis = 0)
    test_table = table.iloc[[row]]
    tf_count = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    tree, tf_count = training(target, train_table, depth, tf_count)
    total_tf_counts['TP'] = total_tf_counts['TP'] + tf_count['TP']
    total_tf_counts['FP'] = total_tf_counts['FP'] + tf_count['FP']
    total_tf_counts['FN'] = total_tf_counts['FN'] + tf_count['FN']
    total_tf_counts['TN'] = total_tf_counts['TN'] + tf_count['TN']

#     print('test_set on row ' + str(row))
    print('training decision tree: ' + str(tree))
    print('training performance' + str(tf_count))
print(total_tf_counts)